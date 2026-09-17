"""最小反向拆分示例：前向照常，反向只先产出输入梯度，参数梯度推迟执行。

它验证的是 Zero Bubble / DualPipe 依赖的那一条最基本机制：
「输入梯度的产生」与「参数梯度的产生」不是同一个完成事件，
因此可以优先传播输入梯度，把参数梯度留到之后执行。

这个脚本不是 DualPipe，也不包含任何 GPU 通信；
它只回答一个问题——先返回 dX、后补 dTheta，数学上是否等价。

运行：
    python torch/dualpipe/codes/01_split_backward_minimal.py
"""

import torch


# 延迟执行的参数梯度任务队列。每一项是 (weight, saved_x, saved_g)。
pending = []


class SplitLinear(torch.autograd.Function):
    """Y = X @ Theta.T，反向只返回 dX，把 dTheta 记入 pending。"""

    @staticmethod
    def forward(ctx, x, theta):
        ctx.save_for_backward(x, theta)
        return x @ theta.T

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        x, theta = ctx.saved_tensors

        # 为稍后的 W 保留 X 与 G；detach 只切断计算图，不释放底层存储。
        pending.append((theta, x.detach(), g.detach()))

        # 现在只返回输入梯度，参数位置返回 None。
        return g @ theta, None


def reference(x, theta, micro_chunks):
    """常规 autograd：完整 batch 一次 backward，作为数值基准。"""
    x_ref = x.detach().clone().requires_grad_()
    theta_ref = theta.detach().clone().requires_grad_()
    loss = (x_ref @ theta_ref.T).square().mean()
    loss.backward()
    return x_ref.grad, theta_ref.grad


def split_backward(x, theta, micro_chunks):
    """拆分版：按 microbatch 前向，反向只产出 dX，最后统一补 dTheta。"""
    x_split = x.detach().clone().requires_grad_()
    theta_split = theta.detach().clone().requires_grad_()

    for micro_x in x_split.chunk(micro_chunks):
        # 与参考实现保持同一归一化口径：整 batch 的平均 loss
        # = 各 microbatch 平均 loss 之和 / microbatch 数。
        loss = SplitLinear.apply(micro_x, theta_split).square().mean() / micro_chunks
        loss.backward()

    assert theta_split.grad is None, "拆分版不应在 backward 中写入参数梯度"

    with torch.no_grad():
        for weight, saved_x, saved_g in pending:
            dw = saved_g.T @ saved_x
            if weight.grad is None:
                weight.grad = dw
            else:
                weight.grad.add_(dw)
        pending.clear()

    return x_split.grad, theta_split.grad


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    x = torch.randn(8, 8, dtype=dtype)
    theta = torch.randn(16, 8, dtype=dtype)
    micro_chunks = 2

    # 观察点一：backward 返回后 dTheta 仍然为空。
    x_probe = x.detach().clone().requires_grad_()
    theta_probe = theta.detach().clone().requires_grad_()
    SplitLinear.apply(x_probe, theta_probe).square().mean().backward()
    dtheta_is_none_after_backward = theta_probe.grad is None
    pending.clear()

    x_grad_ref, theta_grad_ref = reference(x, theta, micro_chunks)
    x_grad_split, theta_grad_split = split_backward(x, theta, micro_chunks)

    dx_err = (x_grad_split - x_grad_ref).abs().max().item()
    dtheta_err = (theta_grad_split - theta_grad_ref).abs().max().item()

    torch.testing.assert_close(x_grad_split, x_grad_ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(theta_grad_split, theta_grad_ref, rtol=1e-10, atol=1e-12)

    print(f"torch                     : {torch.__version__}")
    print(f"dtype                     : {dtype}")
    print(f"x.shape / theta.shape     : {tuple(x.shape)} / {tuple(theta.shape)}")
    print(f"microbatch 数             : {micro_chunks}")
    print(f"backward 返回后 dTheta is None : {dtheta_is_none_after_backward}")
    print(f"max |dX_split - dX_ref|        : {dx_err:.3e}")
    print(f"max |dTheta_split - dTheta_ref|: {dtheta_err:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
