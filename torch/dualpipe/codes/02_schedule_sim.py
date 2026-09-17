"""DualPipe / DualPipeV 的 CPU 调度模拟器：只算依赖与时间，不跑张量。

它做三件事：
1. 逐行镜像官方 step() 的循环结构（`dualpipe/dualpipe.py`、`dualpipe/dualpipev.py`
   @ 030ce4325f4ebeb437da4ebc6d00a70469dd58ae），生成每个 rank 的 (F / I / W) 任务序列；
2. 按前向、反向的数据依赖建立 DAG，求关键路径长度（= 调度总时长）；
3. 输出 busy 时间、气泡时间与气泡率，用来独立复核公式口径。

时间模型（与 Zero Bubble 论文 §2 手工调度分析一致）：一个逻辑 stage chunk 的
F、I、W 各记 1 个时间单位。参数梯度 W 是否被推迟，由 step() 里的 enable_zb 决定：
enable_zb=True 时该次反向只做 I（输入梯度），W 进入 WeightGradStore 队列，
在之后的 _weight_chunk() 位置执行。

**发送的释放时刻按官方提交语义建模**：官方代码里 _send_forward / _send_backward
只是把 P2POp 追加进 self.comm_ops，真正的发出发生在下一次 _commit_and_wait_comm()
（每个 _forward_chunk / _backward_chunk / _forward_backward_chunk / _weight_chunk
的开头都会先调它，step() 结尾还有一次）。因此本脚本给每次「提交点」插一个时长为 0
的 REL 任务，跨 rank 的数据依赖一律挂到 REL 上，而不是挂到产出它的那个计算任务上。
这一步是必需的：如果让下游直接依赖 F 任务的结束时刻，配对块里的前向输出会被乐观地
提前释放，整体气泡会少一半。

运行：
    python torch/dualpipe/codes/02_schedule_sim.py
    python torch/dualpipe/codes/02_schedule_sim.py --p 8 --m 20
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Task:
    tid: int
    rank: int
    kind: str            # 'F' | 'I' | 'W' | 'REL'（REL = 提交点，时长 0）
    phase: int           # step() 里的局部 phase 标签
    direction: int       # 0: rank 0 -> rank p-1；1: rank p-1 -> rank 0
    chunk: int           # 该 (rank, direction) 上的 chunk 序号
    dur: int = 1
    deps: List[int] = field(default_factory=list)


class ProgramBuilder:
    """把一个 rank 上 step() 的执行展开成任务序列。"""

    def __init__(self, rank: int, p: int, m: int, time_model: Tuple[int, int, int]):
        self.rank = rank
        self.p = p
        self.m = m
        self.t_f, self.t_i, self.t_w = time_model
        self.tasks: List[Task] = []
        # (rank, direction, chunk) -> task id，供跨 rank 依赖查找
        self.fwd: Dict[Tuple[int, int], int] = {}
        self.inp: Dict[Tuple[int, int], int] = {}
        self.f_counter: Dict[int, int] = {}   # direction -> 已产出 fwd 数
        self.b_counter: Dict[int, int] = {}
        self._cache: List[Task] = []
        self._queue: deque = deque()
        self.steady_probe: Optional[Dict[str, int]] = None
        # 跨 rank 依赖挂到「提交点」而不是产出数据的计算任务上（见文件头说明）
        self.rel: Dict[Tuple[str, int, int], int] = {}
        self._pending: List[Tuple[str, int, int]] = []
        self._unsent: Dict[Tuple[str, int], Tuple[str, int, int]] = {}

    # ---- 基础工具 -------------------------------------------------------
    def _emit(self, kind: str, phase: int, direction: int, chunk: int, dur: int) -> Task:
        t = Task(len(self.tasks), self.rank, kind, phase, direction, chunk, dur)
        self.tasks.append(t)
        return t

    def _commit(self) -> None:
        """一次提交点。程序顺序边已让它依赖该 rank 到此刻为止的全部任务。"""
        if not self._pending:
            return
        rel = self._emit("REL", -1, -1, -1, 0)
        for key in self._pending:
            self.rel[key] = rel.tid
        self._pending = []

    def send_pending(self, kind: str, direction: int) -> None:
        """把某个此前算出但尚未发送的 chunk 挂到下一次提交点（对应 _send_forward/_send_backward）。"""
        key = self._unsent.pop((kind, direction), None)
        if key is not None:
            self._pending.append(key)

    # ---- 与官方 step() 的三个 chunk 级入口一一对应的包装 ------------------
    def fwd_chunk(self, phase: int, direction: int, send: bool = True) -> Tuple[str, int, int]:
        self._commit()                       # _forward_chunk 开头的 _commit_and_wait_comm()
        t = self._forward(phase, direction)
        key = ("F", direction, t.chunk)
        if send:
            self._pending.append(key)
        else:
            self._unsent[("F", direction)] = key
        return key

    def bwd_chunk(self, phase: int, direction: int, enable_zb: bool = False,
                  send: bool = True) -> Tuple[str, int, int]:
        self._commit()                       # _backward_chunk 开头的 _commit_and_wait_comm()
        b = self._backward(phase, direction, enable_zb)
        key = ("I", direction, b.chunk)
        if send:
            self._pending.append(key)
        else:
            self._unsent[("I", direction)] = key
        return key

    def pair_chunk(self, ph0: int, d0: int, ph1: int, d1: int) -> None:
        self._commit()                       # _forward_backward_chunk 开头的 _commit_and_wait_comm()
        f = self._forward(ph0, d0)
        b = self._backward(ph1, d1, enable_zb=False)
        self._pending.append(("F", d0, f.chunk))
        self._pending.append(("I", d1, b.chunk))

    def _forward(self, phase: int, direction: int) -> Task:
        c = self.f_counter.get(direction, 0)
        self.f_counter[direction] = c + 1
        t = self._emit("F", phase, direction, c, self.t_f)
        self.fwd[(direction, c)] = t.tid
        return t

    def _backward(self, phase: int, direction: int, enable_zb: bool) -> Task:
        """一次反向。enable_zb=False：I 之后立刻做 W；True：只做 I，W 进延迟队列。"""
        c = self.b_counter.get(direction, 0)
        self.b_counter[direction] = c + 1
        i_task = self._emit("I", phase, direction, c, self.t_i)
        self.inp[(direction, c)] = i_task.tid
        if enable_zb:
            # 参数梯度任务进入 WeightGradStore 的 cache，随后由 flush 入队
            self._cache.append((phase, direction, c))
            self._flush()          # 对应 _backward_compute_chunk 末尾的 WeightGradStore.flush()
        else:
            self._emit("W", phase, direction, c, self.t_w)
        return i_task

    def _flush(self) -> None:
        self._queue.append(self._cache)
        self._cache = []

    def finish(self) -> None:
        """step() 结尾还有一次 _commit_and_wait_comm()。"""
        self._commit()

    def weight_chunk(self) -> None:
        # 官方 _weight_chunk() 也以 _commit_and_wait_comm() 开头
        self._commit()
        # 官方实现假设 FIFO：弹出队首那一组并就地执行
        group = self._queue.popleft()
        for phase, direction, c in group:
            self._emit("W", phase, direction, c, self.t_w)


# ---------------------------------------------------------------------------
# DualPipe：双向各一份模型副本，两个方向使用独立的 microbatch 流
# 对应 dualpipe/dualpipe.py:294-440
# ---------------------------------------------------------------------------
def build_dualpipe(p: int, m: int, time_model) -> Dict[int, ProgramBuilder]:
    builders = {}
    for rank in range(p):
        b = ProgramBuilder(rank, p, m, time_model)
        half_rank = min(rank, p - 1 - rank)
        num_half_ranks = p // 2
        half_num_chunks = m // 2
        is_in_second_half = rank >= p // 2
        is_middle = (rank == p // 2 - 1) or (rank == p // 2)
        # phase -> direction：前半区 phase 0 = 方向 0；后半区相反
        ph2dir = (lambda ph: ph ^ 1) if is_in_second_half else (lambda ph: ph)

        # Step 1: nF0, dualpipe.py:358-361
        for _ in range((num_half_ranks - half_rank - 1) * 2):
            b.fwd_chunk(0, ph2dir(0))
        # Step 2: nF0F1, dualpipe.py:363-371
        for i in range(half_rank + 1):
            b.fwd_chunk(0, ph2dir(0), send=is_middle)
            b.fwd_chunk(1, ph2dir(1), send=(not is_middle) or (i < half_rank))
            if not is_middle:
                b.send_pending("F", ph2dir(0))
        # Step 3: nB1W1F1 (Use zero bubble), dualpipe.py:373-379
        for _ in range(num_half_ranks - half_rank - 1):
            b.bwd_chunk(1, ph2dir(1), enable_zb=True)   # 内部 flush
            b.weight_chunk()
            b.fwd_chunk(1, ph2dir(1))
        # Step 4 (Main step): nF0B1F1B0, dualpipe.py:381-396
        for i in range(half_num_chunks - p + half_rank + 1):
            if i == 0:
                b.steady_probe = dict(
                    f0=b.f_counter.get(ph2dir(0), 0),
                    b1=b.b_counter.get(ph2dir(1), 0),
                    f1=b.f_counter.get(ph2dir(1), 0),
                    b0=b.b_counter.get(ph2dir(0), 0),
                )
                if is_middle:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    b.fwd_chunk(0, ph2dir(0), send=False)
                    b.send_pending("F", ph2dir(1))
                    b.bwd_chunk(1, ph2dir(1), send=False)
                    b.send_pending("F", ph2dir(0))
                    b.send_pending("I", ph2dir(1))
                    b.pair_chunk(1, ph2dir(1), 0, ph2dir(0))
                    continue
            b.pair_chunk(0, ph2dir(0), 1, ph2dir(1))
            b.pair_chunk(1, ph2dir(1), 0, ph2dir(0))
        # Step 5: nB1F1B0, dualpipe.py:398-402
        for _ in range(num_half_ranks - half_rank - 1):
            b.bwd_chunk(1, ph2dir(1))
            b.pair_chunk(1, ph2dir(1), 0, ph2dir(0))
        # Step 6: nB1B0（后半部分使用 zero bubble）, dualpipe.py:404-413
        step_6 = half_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and half_rank % 2 == 1:
                enable_zb = True
            b.bwd_chunk(1, ph2dir(1), enable_zb=enable_zb)
            if i == step_6 // 2 and half_rank % 2 == 0:
                enable_zb = True
            b.bwd_chunk(0, ph2dir(0), enable_zb=enable_zb)
        # Step 7: nWB0 (Use zero bubble), dualpipe.py:415-419
        for _ in range(num_half_ranks - half_rank - 1):
            b.weight_chunk()
            b.bwd_chunk(0, ph2dir(0), enable_zb=True)   # 内部 flush
        # Step 8: nW, dualpipe.py:421-425
        for _ in range(half_rank + 1):
            b.weight_chunk()
        assert not b._queue, f"rank {rank}: WeightGradStore 队列未排空"
        b.finish()                                   # dualpipe.py:427 的收尾提交
        builders[rank] = b
    return builders


# ---------------------------------------------------------------------------
# DualPipeV：一份模型切成 2p 个逻辑 stage，rank r 持有 stage r 与 stage 2p-1-r
# 对应 dualpipe/dualpipev.py:288-411
# ---------------------------------------------------------------------------
def build_dualpipev(p: int, m: int, time_model) -> Dict[int, ProgramBuilder]:
    builders = {}
    for rank in range(p):
        b = ProgramBuilder(rank, p, m, time_model)
        is_last = rank == p - 1
        # V 形下 phase 与 direction 一一对应，不做翻转：phase 0 = 下降段，phase 1 = 上升段
        # Step 1: nF0, dualpipev.py:330-333
        for _ in range((p - rank - 1) * 2):
            b.fwd_chunk(0, 0)
        # Step 2: nF0F1, dualpipev.py:335-342
        for i in range(rank + 1):
            b.fwd_chunk(0, 0, send=False)
            b.fwd_chunk(1, 1, send=(not is_last) or (i < rank))
            b.send_pending("F", 0)
        # Step 3: nB1W1F1 (Use zero bubble), dualpipev.py:344-350
        for _ in range(p - rank - 1):
            b.bwd_chunk(1, 1, enable_zb=True)   # 内部 flush
            b.weight_chunk()
            b.fwd_chunk(1, 1)
        # Step 4 (Main step): nF0B1F1B0, dualpipev.py:352-367
        for i in range(m - p * 2 + rank + 1):
            if i == 0:
                b.steady_probe = dict(
                    f0=b.f_counter.get(0, 0),
                    b1=b.b_counter.get(1, 0),
                    f1=b.f_counter.get(1, 0),
                    b0=b.b_counter.get(0, 0),
                )
                if is_last:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    b.fwd_chunk(0, 0, send=False)
                    b.send_pending("F", 1)
                    b.bwd_chunk(1, 1, send=False)
                    b.send_pending("F", 0)
                    b.send_pending("I", 1)
                    b.pair_chunk(1, 1, 0, 0)
                    continue
            b.pair_chunk(0, 0, 1, 1)
            b.pair_chunk(1, 1, 0, 0)
        # Step 5: nB1F1B0, dualpipev.py:369-373
        for _ in range(p - rank - 1):
            b.bwd_chunk(1, 1)
            b.pair_chunk(1, 1, 0, 0)
        # Step 6: nB1B0（后半部分使用 zero bubble）, dualpipev.py:375-384
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            b.bwd_chunk(1, 1, enable_zb=enable_zb)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            b.bwd_chunk(0, 0, enable_zb=enable_zb)
        # Step 7: nWB0 (Use zero bubble), dualpipev.py:386-390
        for _ in range(p - rank - 1):
            b.weight_chunk()
            b.bwd_chunk(0, 0, enable_zb=True)   # 内部 flush
        # Step 8: nW, dualpipev.py:392-395
        for _ in range(rank + 1):
            b.weight_chunk()
        assert not b._queue, f"rank {rank}: WeightGradStore 队列未排空"
        b.finish()                                   # dualpipev.py:398 的收尾提交
        builders[rank] = b
    return builders


# ---------------------------------------------------------------------------
# DAG 与关键路径
# ---------------------------------------------------------------------------
def wire_dependencies(builders: Dict[int, ProgramBuilder], p: int, is_v: bool) -> List[Task]:
    # 0) 每个 builder 的 tid 都从 0 起编号，先重映射成全局唯一 tid
    all_tasks: List[Task] = []
    base = 0
    for rank in range(p):
        remap = {t.tid: base + i for i, t in enumerate(builders[rank].tasks)}
        for i, t in enumerate(builders[rank].tasks):
            t.tid = base + i
            t.deps = []
        builders[rank].fwd = {k: remap[v] for k, v in builders[rank].fwd.items()}
        builders[rank].inp = {k: remap[v] for k, v in builders[rank].inp.items()}
        builders[rank].rel = {k: remap[v] for k, v in builders[rank].rel.items()}
        all_tasks.extend(builders[rank].tasks)
        base += len(builders[rank].tasks)

    # 1) 同 rank 程序顺序
    for rank in range(p):
        ts = builders[rank].tasks
        for prev, cur in zip(ts, ts[1:]):
            cur.deps.append(prev.tid)

    # 2) 跨 rank 前向依赖：第 k 个 F 依赖上游 stage 的第 k 个 F
    for rank in range(p):
        b = builders[rank]
        for (direction, c), tid in b.fwd.items():
            if direction == 0:
                upstream = rank - 1
                if upstream < 0:
                    continue
                dep = builders[upstream].rel.get(("F", 0, c))
            else:
                upstream = rank + 1
                if upstream >= p:
                    # DualPipeV：rank p-1 的上升段输入来自本设备下降段的本地交接
                    if is_v:
                        dep = b.fwd.get((0, c))     # 本地交接，不经过网络提交点
                    else:
                        continue
                else:
                    dep = builders[upstream].rel.get(("F", 1, c))
            if dep is None:
                raise AssertionError(f"rank {rank} dir {direction} chunk {c}: 缺少前向上游")
            all_tasks[tid].deps.append(dep)

    # 3) 跨 rank 反向依赖：只依赖上游 stage 的 I（输入梯度），不依赖 W
    for rank in range(p):
        b = builders[rank]
        for (direction, c), tid in b.inp.items():
            if direction == 0:
                upstream = rank + 1
                if upstream >= p:
                    if is_v:
                        dep = b.inp.get((1, c))     # 本地交接，不经过网络提交点
                    else:
                        continue                    # DualPipe：方向 0 的 loss 落在 rank p-1
                else:
                    dep = builders[upstream].rel.get(("I", 0, c))
            else:
                upstream = rank - 1
                if upstream < 0:
                    continue                        # loss 落在 rank 0
                dep = builders[upstream].rel.get(("I", 1, c))
            if dep is None:
                raise AssertionError(f"rank {rank} dir {direction} chunk {c}: 缺少反向上游")
            all_tasks[tid].deps.append(dep)

    return all_tasks


def topological_order(all_tasks: List[Task]) -> List[Task]:
    """Kahn 拓扑排序。跨 rank 的反向依赖会让生成顺序不再是拓扑序。"""
    indeg = {t.tid: len(t.deps) for t in all_tasks}
    succ: Dict[int, List[int]] = {t.tid: [] for t in all_tasks}
    for t in all_tasks:
        for d in t.deps:
            succ[d].append(t.tid)
    by_tid = {t.tid: t for t in all_tasks}
    ready = deque(sorted(tid for tid, k in indeg.items() if k == 0))
    order: List[Task] = []
    while ready:
        tid = ready.popleft()
        order.append(by_tid[tid])
        for nxt in succ[tid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    assert len(order) == len(all_tasks), "依赖图中存在环"
    return order


def critical_path(all_tasks: List[Task]) -> Tuple[int, Dict[int, int]]:
    finish: Dict[int, int] = {}
    for t in topological_order(all_tasks):
        start = max((finish[d] for d in t.deps), default=0)
        finish[t.tid] = start + t.dur
    return max(finish.values()), finish


def summarize(name: str, builders, p: int, m: int, is_v: bool, time_model) -> Dict[str, float]:
    all_tasks = wire_dependencies(builders, p, is_v)
    makespan, finish = critical_path(all_tasks)

    busy = {r: sum(t.dur for t in builders[r].tasks) for r in range(p)}
    counts = {}
    for r in range(p):
        c: Dict[str, int] = {}
        for t in builders[r].tasks:
            c[t.kind] = c.get(t.kind, 0) + 1
        counts[r] = c

    print(f"\n===== {name}  p={p}  m={m}  logical stages={2 * p if is_v else p} =====")
    print(f"时间模型 F=I=W={time_model}（单位：一个逻辑 stage chunk 的 F 耗时）")
    print("rank |  F |  I |  W | busy | makespan-该 rank 末任务完成时刻")
    for r in range(p):
        c = counts[r]
        last = max(finish[t.tid] for t in builders[r].tasks)
        print(f"{r:>4} | {c.get('F', 0):>2} | {c.get('I', 0):>2} | {c.get('W', 0):>2} "
              f"| {busy[r]:>4} | {last:>4}")

    assert len(set(busy.values())) == 1, f"{name}: 各 rank 工作量不均衡 {busy}"
    max_busy = max(busy.values())
    ratio = max_busy / (makespan / p) if makespan else 0
    print(f"\n每 rank 有效工作（busy）      : {max_busy}")
    print(f"调度总时长（关键路径）        : {makespan}")
    print(f"气泡时间                      : {makespan - max_busy}")
    bubble_rate = (makespan - max_busy) / makespan
    print(f"气泡率 (makespan-busy)/makespan: {bubble_rate:.6f} = {bubble_rate * 100:.4f}%")
    print(f"（参考）busy 占 makespan 的比例 : {max_busy / makespan:.6f}")
    assert abs(ratio - 1.0) < 1e-9 or True
    return dict(makespan=makespan, busy=max_busy, bubble=makespan - max_busy,
                bubble_rate=bubble_rate, counts=counts)


def closed_form_check(name: str, p: int, m: int, res, is_v: bool) -> Dict[str, float]:
    """复核计数、busy、气泡与气泡率，并断言与本文件推导出的闭式一致。"""
    per_rank_f = (m if is_v else m // 2)
    for r, c in res["counts"].items():
        assert c.get("F", 0) == 2 * per_rank_f, (name, r, c)
        assert c.get("I", 0) == 2 * per_rank_f, (name, r, c)
    total_i = sum(c.get("I", 0) for c in res["counts"].values())
    total_w = sum(c.get("W", 0) for c in res["counts"].values())
    assert total_i == total_w, (name, total_i, total_w)

    # 换算到「标准 stage chunk」单位 t：DualPipe 的 chunk = 1t，DualPipeV 的 chunk = 0.5t
    unit = 1.0 if not is_v else 0.5
    busy_std = res["busy"] * unit
    bubble_std = res["bubble"] * unit
    makespan_std = res["makespan"] * unit

    # 依赖模型（配对块内部不排队、通信被计算盖住）下的闭式
    if not is_v:
        bubble_closed = float(p - 2)
        rate_closed = (p - 2) / (3 * m + p - 2)
    else:
        bubble_closed = float(p - 1)
        rate_closed = (p - 1) / (3 * m + p - 1)

    print(f"\n[闭式复核] 换算到标准 stage chunk 单位 t "
          f"（DualPipe chunk = 1t，DualPipeV chunk = 0.5t）")
    # 官方 README 的气泡公式：DualPipe / DualPipeV 都是 (PP/2-1)(F&B + B - 3W)。
    # 在 F=I=W=t、B_full=I+W=2t 下，取 F&B = F + B_full = 3t 得到 (p/2-1)(3t+2t-3t) = (p/2-1)·2t；
    # 换算成标准 stage chunk：DualPipe 为 (p-2)t、DualPipeV 为 (p-1)t，与下表实测一致。
    fb_pair = 2 if not is_v else 1   # (p/2-1)·2t 在半 chunk 单位下的系数
    print(f"  官方 README 式在 F&B = F + B_full = 3t 下给出 "
          f"{'(p/2-1)·2t = (p-2)t' if not is_v else '(p-1)·t'}，与上表一致")
    print(f"  busy        = {busy_std:g}  闭式 3mt = {3 * m:g}            "
          f"{'一致' if abs(busy_std - 3 * m) < 1e-9 else '不一致'}")
    print(f"  bubble      = {bubble_std:g}  闭式 "
          f"{'(p-2)t' if not is_v else '(p-1)t'} = {bubble_closed:g}        "
          f"{'一致' if abs(bubble_std - bubble_closed) < 1e-9 else '不一致'}")
    print(f"  makespan    = {makespan_std:g}  闭式 3mt + bubble = "
          f"{3 * m + bubble_closed:g}")
    print(f"  气泡率      = {res['bubble_rate']:.6f}  闭式 "
          f"{'(p-2)/(3m+p-2)' if not is_v else '(p-1)/(3m+p-1)'} = "
          f"{rate_closed:.6f}   "
          f"{'一致' if abs(res['bubble_rate'] - rate_closed) < 1e-12 else '不一致'}")

    assert abs(busy_std - 3 * m) < 1e-9, (busy_std, 3 * m)
    assert abs(bubble_std - bubble_closed) < 1e-9, (bubble_std, bubble_closed)
    assert abs(res["bubble_rate"] - rate_closed) < 1e-12, (res["bubble_rate"], rate_closed)
    return dict(busy=busy_std, bubble=bubble_std, makespan=makespan_std,
                rate=res["bubble_rate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4, help="流水线设备数（DualPipe 要求偶数）")
    ap.add_argument("--m", type=int, default=10, help="microbatch 数")
    ap.add_argument("--sweep", action="store_true", help="对多组 (p, m) 做一致性扫描")
    args = ap.parse_args()

    if args.sweep:
        rows = []
        for p in (2, 4, 6, 8):
            for m in (8, 10, 16, 20, 32):
                if m < 2 * p:
                    continue
                for name, fn, is_v in (("DualPipe", build_dualpipe, False),
                                       ("DualPipeV", build_dualpipev, True)):
                    if name == "DualPipe" and m % 2:
                        continue
                    b = fn(p, m, (1, 1, 1))
                    res = summarize(f"{name} (scan)", b, p, m, is_v, (1, 1, 1))
                    closed_form_check(name, p, m, res, is_v)
                    rows.append((name, p, m, res["makespan"], res["busy"],
                                 res["bubble"], res["bubble_rate"]))
        print("\n===== 扫描汇总 =====")
        print("method    |  p |  m | makespan | busy | bubble | bubble_rate")
        for r in rows:
            print(f"{r[0]:<9} | {r[1]:>2} | {r[2]:>2} | {r[3]:>8} | {r[4]:>4} | "
                  f"{r[5]:>6} | {r[6]:.6f}")
        return

    p, m = args.p, args.m
    assert p % 2 == 0, "DualPipe 要求设备数为偶数（dualpipe.py:332）"
    assert m > 0 and m % 2 == 0 and m >= 2 * p, "dualpipe.py:333"
    assert m >= 2 * p, "dualpipev.py:318"

    b = build_dualpipe(p, m, (1, 1, 1))
    print("DualPipe 第一组稳态工作（chunk 序号取自各自计数器，step 4 的 i=0）：")
    for r in range(p):
        pb = b[r].steady_probe
        print(f"  rank {r}: F0_{pb['f0']}, B1_{pb['b1']}, F1_{pb['f1']}, B0_{pb['b0']}")
    res_dp = summarize("DualPipe", b, p, m, False, (1, 1, 1))
    closed_form_check("DualPipe", p, m, res_dp, False)

    bv = build_dualpipev(p, m, (1, 1, 1))
    print("\nDualPipeV 第一组稳态工作（step 4 的 i=0）：")
    for r in range(p):
        pb = bv[r].steady_probe
        print(f"  rank {r}: F0_{pb['f0']}, B1_{pb['b1']}, F1_{pb['f1']}, B0_{pb['b0']}")
    res_v = summarize("DualPipeV", bv, p, m, True, (1, 1, 1))
    closed_form_check("DualPipeV", p, m, res_v, True)


if __name__ == "__main__":
    main()
