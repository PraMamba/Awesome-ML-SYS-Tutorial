"""教学实现：单进程 MoE 路由、assignment 展开与加权归约。

这不是 DeepEP 源码，也不依赖 DeepEP。它的唯一用途是把文章第 2、3 章用到的语义
显式写出来：token->expert 的路由、assignment 展开、按专家排序、以及路由权重
到底在哪一步乘上去。

运行：
    python minimal_moe.py
"""

import torch


def moe_by_expert(x, expert_ids, probs, weights):
    """按专家展开的 MoE 前向。

    x:          [T, H]
    expert_ids: [T, K]，int64，-1 表示该路由无效
    probs:      [T, K]
    weights:    [E, H, H]

    返回 y: [T, H]
    """
    t_count, k_count = expert_ids.shape
    expert_count = weights.shape[0]

    # 1) 把 [T, K] 的路由展开成 assignment 列表，每个有效 token-expert 对占一行。
    token = torch.arange(t_count, device=x.device).repeat_interleave(k_count)
    expert = expert_ids.reshape(-1)
    prob = probs.reshape(-1)
    valid = expert >= 0
    token, expert, prob = token[valid], expert[valid], prob[valid]

    # 2) 真实系统需要让同一专家的输入拼成连续矩阵，所以先按专家做稳定排序。
    order = torch.argsort(expert, stable=True)
    token, expert, prob = token[order], expert[order], prob[order]

    # 3) dispatch 的语义：每个 assignment 取一份输入副本。
    dispatched = x.index_select(0, token)

    # 4) 逐专家计算 tanh(x_e @ W_e)，这里用最朴素的循环代替 grouped GEMM。
    pieces = []
    for e in range(expert_count):
        rows = dispatched[expert == e]
        pieces.append(torch.tanh(rows @ weights[e]))
    expert_output = torch.cat(pieces, dim=0)

    # 5) 路由权重在这里乘一次，然后按来源 token 归约回原顺序。
    out = torch.zeros_like(x)
    return out.index_add(0, token, expert_output * prob[:, None])


def moe_reference(x, expert_ids, probs, weights):
    """逐 token、逐路由的朴素参考实现，用来校验展开与归约是否正确。"""
    t_count, k_count = expert_ids.shape
    # 用 x[t] * 0 而不是 zeros_like：没有任何有效路由时，输出仍然是 x 的函数
    # （梯度为 0），这样参考实现和展开实现在"全空"用例下也能比较梯度。
    acc = [x[t] * 0.0 for t in range(t_count)]
    for t in range(t_count):
        for k in range(k_count):
            e = int(expert_ids[t, k])
            if e < 0:
                continue
            acc[t] = acc[t] + probs[t, k] * torch.tanh(x[t] @ weights[e])
    return torch.stack(acc, dim=0)


def _make_case(ids, seed=7, t_count=4, hidden=8, topk=2, expert_count=8):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(t_count, hidden, generator=generator, dtype=torch.float64)
    probs = torch.randn(t_count, topk, generator=generator, dtype=torch.float64).softmax(-1)
    weights = torch.randn(expert_count, hidden, hidden, generator=generator, dtype=torch.float64)
    return x, probs, weights


def _forward_and_grad(x, ids, probs, weights):
    x = x.clone().requires_grad_()
    probs = probs.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    y = moe_by_expert(x, ids, probs, weights)
    y.square().sum().backward()
    return y.detach(), x.grad, probs.grad, weights.grad


def _reference_and_grad(x, ids, probs, weights):
    x = x.clone().requires_grad_()
    probs = probs.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    y = moe_reference(x, ids, probs, weights)
    y.square().sum().backward()
    return y.detach(), x.grad, probs.grad, weights.grad


def check_case(name, ids):
    x, probs, weights = _make_case(ids)
    got = _forward_and_grad(x, ids, probs, weights)
    want = _reference_and_grad(x, ids, probs, weights)
    labels = ["y", "dX", "dProbs", "dWeights"]
    worst = 0.0
    for got_item, want_item in zip(got, want):
        if got_item is None or want_item is None:
            continue
        worst = max(worst, float((got_item - want_item).abs().max()))
    status = "PASS" if worst < 1e-10 else "FAIL"
    print(f"[{status}] {name}: 前向与三项梯度的最大绝对误差 = {worst:.3e}")
    return status == "PASS"


def weight_placement_demo():
    """复算文章里那个一维例子：权重乘在哪里会改变结果。"""
    z1, z2 = torch.tensor(10.0), torch.tensor(20.0)
    p1, p2 = torch.tensor(0.8), torch.tensor(0.2)
    correct = p1 * z1 + p2 * z2
    missing = z1 + z2
    doubled = p1 * p1 * z1 + p2 * p2 * z2
    print(f"正确：0.8*10 + 0.2*20 = {float(correct):.1f}")
    print(f"漏乘权重：10 + 20 = {float(missing):.1f}")
    print(f"重复乘权重：0.8^2*10 + 0.2^2*20 = {float(doubled):.1f}")


def main():
    ids_normal = torch.tensor([[0, 1], [1, 2], [4, 6], [6, 7]])
    ids_with_invalid = torch.tensor([[0, -1], [1, 2], [-1, 6], [3, 7]])
    ids_single_token = torch.tensor([[5, 5], [-1, -1], [-1, -1], [-1, -1]])
    ids_all_invalid = torch.full((4, 2), -1)

    ok = True
    ok &= check_case("常规路由（专家 3、5 无人选中）", ids_normal)
    ok &= check_case("含无效路由 (-1)", ids_with_invalid)
    ok &= check_case("单个 token 命中同一专家两次", ids_single_token)
    ok &= check_case("所有路由均无效", ids_all_invalid)

    # 逻辑展开量与 rank 去重量的差别：文章的 4 rank / 8 expert 例子。
    ids_group = torch.tensor([[0, 1], [1, 2], [4, 6], [6, 7]])
    local_experts = 2
    assignments = ids_group.reshape(-1)
    dest = assignments // local_experts
    token = torch.arange(ids_group.shape[0]).repeat_interleave(ids_group.shape[1])
    pairs = token * 4 + dest  # (token, 目的 rank) 对的编码
    print(f"token-expert assignment = {assignments.numel()}，"
          f"按 (token, 目的 rank) 去重后 = {pairs.unique().numel()}，"
          f"涉及的目的 rank = {dest.unique().numel()}")
    print("网络上传的份数对应第二个数：同一 token 命中同一 rank 的多个专家只传一份。")

    weight_placement_demo()
    print("全部用例通过" if ok else "存在失败用例")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
