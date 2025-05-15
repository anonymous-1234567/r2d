import torch

def spectral_radius_hessian(args, loss, model, max_iters = 1000, tol = 1e-7):
    """
    Power‐iteration to estimate the spectral radius of the Hessian of `loss`.
    Args:
        args.seed       (int, optional): random seed for reproducibility
        tol        (float): convergence tolerance on eigenvalue change
    Inputs:
        loss  – a scalar torch.Tensor (typically the batch‐average loss)
        model – a torch.nn.Module whose parameters define the Hessian
    Returns:
        eig (torch.Tensor) – estimated top |eigenvalue| of Hessian
    """
    # 1) optional seeding
    if hasattr(args, 'seed') and args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 2) first‐order grads (create_graph=True so we can do Hessian‐vector products)
    params = list(model.parameters())
    grad_params = torch.autograd.grad(
        loss, params, create_graph=True
    )

    # 3) flatten parameter shapes & prep a unit‐random initial vector v
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]
    total  = sum(sizes)

    device = loss.device
    v = torch.randn(total, device=device)
    v = v / v.norm()

    prev_eig = None
    for _ in range(max_iters):
        # split v into per‐param tensors
        vs = torch.split(v, sizes)
        v_tensors = [v_i.view(shape) for v_i, shape in zip(vs, shapes)]

        # 4) Hessian-vector product H v
        hv = torch.autograd.grad(
            grad_params, params,
            grad_outputs=v_tensors,
            retain_graph=True
        )
        hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])

        # 5) Rayleigh quotient => eigenvalue estimate
        eig = torch.dot(v, hv_flat)

        # 6) next iterate v ← H v / ∥H v∥
        v = hv_flat / (hv_flat.norm() + 1e-12)
        v = v.detach()  # drop graph to avoid memory growth

        # 7) convergence check
        if prev_eig is not None and torch.abs(eig - prev_eig) < tol:
            break
        prev_eig = eig

    return eig


def spectral_radius(args, loss, model, max_iters = 1000, tol = 1e-7):
    """
    Power‐iteration to estimate the spectral radius of (I – eta*H).
    Args:
        args.seed       (int, optional): random seed
        max_iters  (int): max power‐iterations
        tol        (float): tolerance on eigenvalue change
        args.lr        (float): the η scalar in I – η H
    Inputs:
        loss  – a scalar torch.Tensor
        model – torch.nn.Module
    Returns:
        eig (torch.Tensor) – estimated top |eigenvalue| of (I – η H)
    """
    # 1) optional seeding
    if hasattr(args, 'seed') and args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 2) first‐order grads
    params = list(model.parameters())
    grad_params = torch.autograd.grad(
        loss, params, create_graph=True
    )

    # 3) prep shapes and init v
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]
    total  = sum(sizes)

    device = loss.device
    v = torch.randn(total, device=device)
    v = v / v.norm()

    prev_eig = None
    for _ in range(max_iters):
        vs = torch.split(v, sizes)
        v_tensors = [v_i.view(shape) for v_i, shape in zip(vs, shapes)]

        # 4) H v
        hv = torch.autograd.grad(
            grad_params, params,
            grad_outputs=v_tensors,
            retain_graph=True
        )
        hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])

        # 5) form (I – η H) v = v − η (H v)
        m_v = v - args.lr * hv_flat

        # 6) Rayleigh quotient
        eig = torch.dot(v, m_v)

        # 7) normalize for next iterate
        v = m_v / (m_v.norm() + 1e-12)
        v = v.detach()

        # 8) convergence
        if prev_eig is not None and torch.abs(eig - prev_eig) < tol:
            break
        prev_eig = eig

    return eig
