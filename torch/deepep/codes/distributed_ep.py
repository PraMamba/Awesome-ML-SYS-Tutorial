"""教学实现：用不等长的 all_to_all_single 走一遍 EP 的 dispatch / combine。

运行：
    torchrun --standalone --nproc-per-node=4 distributed_ep.py

这是教学代码，不是 DeepEP 源码，也不调用 DeepEP。它用 Gloo + CPU 张量验证的是
dispatch/combine 的语义与梯度，不测量 RDMA / NVLink 吞吐，也不涉及 DeepEP 的
CUDA 内核、布局或同步策略。

它要演示的三件事：
1. 每个 token-expert assignment 先按目的 rank 排序，再交换分片长度；
2. 反向交换的分片长度刚好对调（combine 的 backward 就是 dispatch）；
3. 路由权重只在源 rank 乘一次，且乘在哪一步会直接改变结果。
"""

import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    """把不等长 all_to_all_single 包装成一个可求导算子。

    forward 按 (out_sizes, in_sizes) 发送；backward 把两者对调。
    这就是训练框架里 dispatch/combine 互为反向的通信契约来源。
    """

    @staticmethod
    def forward(ctx, x, out_sizes, in_sizes):
        ctx.out_sizes, ctx.in_sizes = out_sizes, in_sizes
        out = x.new_empty((sum(out_sizes), x.shape[1]))
        dist.all_to_all_single(
            out, x.contiguous(), output_split_sizes=out_sizes, input_split_sizes=in_sizes
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty((sum(ctx.in_sizes), grad_out.shape[1]))
        dist.all_to_all_single(
            grad_in,
            grad_out.contiguous(),
            output_split_sizes=ctx.in_sizes,
            input_split_sizes=ctx.out_sizes,
        )
        return grad_in, None, None


def _exchange_int(values, out_sizes, in_sizes):
    """整数元数据（专家编号）的交换，不参与求导。"""
    out = torch.empty(sum(out_sizes), dtype=values.dtype)
    dist.all_to_all_single(
        out, values.contiguous(), output_split_sizes=out_sizes, input_split_sizes=in_sizes
    )
    return out


def distributed_moe(x, expert_ids, probs, weights, local_experts, world_size, rank):
    """一次完整的本地路由 -> dispatch -> 专家计算 -> combine。

    x:          [T, H]
    expert_ids: [T, K]，全局专家编号，-1 表示无效路由
    probs:      [T, K]
    weights:    [E_local, H, H]，本 rank 持有的专家参数
    """
    t_count, k_count = expert_ids.shape

    # --- 1) 本地路由：展开 assignment，算出目的 rank，按目的 rank 稳定排序 ---
    token = torch.arange(t_count).repeat_interleave(k_count)
    expert = expert_ids.reshape(-1)
    prob = probs.reshape(-1)
    valid = expert >= 0
    token, expert, prob = token[valid], expert[valid], prob[valid]
    dest = expert // local_experts

    order = torch.argsort(dest, stable=True)
    token, expert, prob, dest = (t[order] for t in (token, expert, prob, dest))
    send_sizes = torch.bincount(dest, minlength=world_size).tolist()

    # --- 2) 交换计数：每个目的 rank 需要知道会收到多少行 ---
    send_counts = torch.tensor(send_sizes, dtype=torch.long)
    recv_counts = torch.empty(world_size, dtype=torch.long)
    dist.all_to_all_single(recv_counts, send_counts)
    recv_sizes = recv_counts.tolist()

    # --- 3) dispatch：发送输入副本与专家编号 ---
    send_x = x.index_select(0, token)
    recv_x = _AllToAll.apply(send_x, recv_sizes, send_sizes)
    recv_expert = _exchange_int(expert, recv_sizes, send_sizes)

    # --- 4) 本地专家计算：按专家排序 -> 计算 -> 逆排列回通信顺序 ---
    local_index = recv_expert - rank * local_experts
    local_order = torch.argsort(local_index, stable=True)
    inverse = torch.empty_like(local_order)
    inverse[local_order] = torch.arange(local_order.numel())
    sorted_x = recv_x.index_select(0, local_order)
    sorted_local = local_index.index_select(0, local_order)
    pieces = [
        torch.tanh(sorted_x[sorted_local == i] @ weights[i]) for i in range(local_experts)
    ]
    sorted_out = torch.cat(pieces, dim=0)
    computed = sorted_out.index_select(0, inverse)

    # --- 5) combine：把专家输出换回源 rank，分片长度对调 ---
    back = _AllToAll.apply(computed, send_sizes, recv_sizes)

    # --- 6) 源 rank 乘一次路由权重，再按来源 token 归约 ---
    out = torch.zeros_like(x)
    y = out.index_add(0, token, back * prob[:, None])
    pair_code = token * world_size + dest
    stats = {
        "assignments": int(token.numel()),
        "rows_sent": int(sum(send_sizes)),
        "rows_if_dedup": int(pair_code.unique().numel()),
        "rows_received": int(sum(recv_sizes)),
        "nonzero_send_chunks": int(sum(1 for s in send_sizes if s > 0)),
    }
    return y, stats


# ---------------------------------------------------------------------------
# 校验：把 4 个 rank 的输入拼成全局问题，用单进程参考实现对照
# ---------------------------------------------------------------------------


def moe_reference(x, expert_ids, probs, weights):
    """与 minimal_moe.py 相同的参考实现（逐 token、逐路由累加）。"""
    t_count, k_count = expert_ids.shape
    acc = [x[t] * 0.0 for t in range(t_count)]
    for t in range(t_count):
        for k in range(k_count):
            e = int(expert_ids[t, k])
            if e < 0:
                continue
            acc[t] = acc[t] + probs[t, k] * torch.tanh(x[t] @ weights[e])
    return torch.stack(acc, dim=0)


def _gather(value):
    buckets = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(buckets, value.contiguous())
    return torch.cat(buckets, dim=0)


def _diff(got, want):
    """对比两个梯度；某一侧没有参与计算时 PyTorch 会给出 None，按全零处理。"""
    if got is None and want is None:
        return 0.0
    if got is None:
        got = torch.zeros_like(want)
    if want is None:
        want = torch.zeros_like(got)
    return float((got.detach() - want.detach()).abs().max())


def run_case(name, ids_fn, hidden=8, local_experts=2, t_count=4, topk=2):
    world_size, rank = dist.get_world_size(), dist.get_rank()
    expert_count = local_experts * world_size
    generator = torch.Generator().manual_seed(7 + rank)

    x = torch.randn(t_count, hidden, generator=generator, dtype=torch.float64)
    probs = torch.randn(t_count, topk, generator=generator, dtype=torch.float64).softmax(-1)
    weights = torch.randn(
        local_experts, hidden, hidden, generator=generator, dtype=torch.float64
    )
    ids = ids_fn(rank, t_count, topk, expert_count)

    x_leaf = x.clone().requires_grad_()
    probs_leaf = probs.clone().requires_grad_()
    weights_leaf = weights.clone().requires_grad_()
    y, stats = distributed_moe(
        x_leaf, ids, probs_leaf, weights_leaf, local_experts, world_size, rank
    )
    y.square().sum().backward()

    # 参考实现：全局 token / 全局路由 / 全局专家参数
    x_all, ids_all, probs_all, weights_all = (
        _gather(x),
        _gather(ids),
        _gather(probs),
        _gather(weights),
    )
    x_ref = x_all.clone().requires_grad_()
    probs_ref = probs_all.clone().requires_grad_()
    weights_ref = weights_all.clone().requires_grad_()
    y_ref = moe_reference(x_ref, ids_all, probs_ref, weights_ref)
    y_ref.square().sum().backward()

    begin, end = rank * t_count, (rank + 1) * t_count
    errors = {
        "y": float((y.detach() - y_ref[begin:end].detach()).abs().max()),
        "dX": _diff(x_leaf.grad, x_ref.grad[begin:end] if x_ref.grad is not None else None),
        "dProbs": _diff(
            probs_leaf.grad,
            probs_ref.grad[begin:end] if probs_ref.grad is not None else None,
        ),
        "dWeights": _diff(
            weights_leaf.grad,
            (
                weights_ref.grad[rank * local_experts : (rank + 1) * local_experts]
                if weights_ref.grad is not None
                else None
            ),
        ),
    }
    worst = torch.tensor(max(errors.values()), dtype=torch.float64)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)

    if rank == 0:
        status = "PASS" if float(worst) < 1e-10 else "FAIL"
        detail = " ".join(f"{k}={v:.1e}" for k, v in errors.items())
        print(f"[{status}] {name}: 最大绝对误差 {float(worst):.3e} ({detail})")
        print(
            f"        rank0 统计：assignment={stats['assignments']}，"
            f"本实现发送行数={stats['rows_sent']}，"
            f"按 (token,目的rank) 去重后本可为 {stats['rows_if_dedup']} 行，"
            f"接收行数={stats['rows_received']}"
        )
    return float(worst) < 1e-10


def main():
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 4:
        if rank == 0:
            print(f"本脚本按 4 个 rank 设计（当前 {world_size}），请用 --nproc-per-node=4")
        dist.destroy_process_group()
        return 1

    expert_count = 2 * world_size
    base = torch.tensor([[0, 1], [1, 2], [4, 6], [6, 7]])

    def ids_shifted(r, t_count, topk, n_experts):
        # 全局避开最后一个专家，用来覆盖"某个专家没人选中"的情况
        return (base + r) % (n_experts - 1)

    def ids_one_rank_empty(r, t_count, topk, n_experts):
        if r == 1:
            return torch.full((t_count, topk), -1)
        return (base + r) % (n_experts - 1)

    def ids_same_pair(r, t_count, topk, n_experts):
        # 每个 token 同时命中同一 rank 上的两个专家：assignment 8 份，实际只需 4 份
        local = torch.tensor([[2 * r, 2 * r + 1]])
        return local.repeat(t_count, 1)

    def ids_all_empty(r, t_count, topk, n_experts):
        return torch.full((t_count, topk), -1)

    results = [
        run_case("常规路由（有一个专家全局无人选中）", ids_shifted),
        run_case("某个 rank 完全没有 token", ids_one_rank_empty),
        run_case("同一 token 命中同一 rank 的两个专家（去重）", ids_same_pair),
        run_case("所有 rank 都没有 token", ids_all_empty),
    ]
    ok = all(results)
    if rank == 0:
        print("全部用例通过" if ok else "存在失败用例")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
