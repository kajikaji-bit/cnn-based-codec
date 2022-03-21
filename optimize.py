import torch


def optimize(pmodel, x, max_iter=30, lr=1.0, alpha=1.0):
    params = [param.values.requires_grad_(True) for param in pmodel.model_params]
    init_params = [param.init_param.reshape(-1, 1) for param in pmodel.model_params]
    optimizer = torch.optim.LBFGS(params, lr=lr, line_search_fn="strong_wolfe")
    min_cost = float("inf")

    def closure():
        optimizer.zero_grad()
        cost = 0.
        cost += pmodel.log_prob(x).sum()
        for param, init_param in zip(params, init_params):
            cost += alpha * ((param - init_param) ** 2).sum()
        cost.backward()
        return cost

    for i in range(max_iter):
        optimizer.step(closure)
        cost = closure() + 8 * sum([param.nelement() for param in params])

        print(f"STEP-{i} {round(cost.item())} ({round(cost.item() / x.nelement(), 3)} b/p)")
        for param in params:
            print(f"param:\n{param.detach().numpy()}")
            print(f"grad:\n{param.grad.numpy()}")

        if cost >= min_cost:
            break
        else:
            min_cost = cost.detach().clone()


def selective_optimize(pmodel, x, max_iter=30, lr=1.0, alpha=1.0):
    # 最適化
    optimize(pmodel, x, max_iter, lr, alpha)

    # 結合係数が0.001以下の分布を削除
    weight = pmodel.mixture_weights().reshape(-1, len(pmodel.flag)).mean(axis=0)
    pmodel.flag.copy_(weight > 0.001)

    # 再度最適化
    optimize(pmodel, x, max_iter, lr, alpha)
