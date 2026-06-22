# quat_utils.py (clean)
import torch


# q = [w, x, y, z]


def q_normalize(q, eps=1e-12):
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


def q_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def q_inv(q, eps=1e-12):
    return q_conj(q) / (torch.sum(q*q, dim=-1, keepdim=True) + eps)


def q_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


def q_exp_pure(v):
    """
    exp([0, v]) , v in R^3
    """
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)

    sinc = torch.where(
        theta > 1e-8,
        torch.sin(theta) / theta,
        torch.ones_like(theta)
    )

    return torch.cat([
        torch.cos(theta),
        sinc * v
    ], dim=-1)


def q_log(q):
    """
    log(q) -> R^3
    """
    q = q_normalize(q)

    w = q[..., :1]
    v = q[..., 1:]

    vnorm = torch.linalg.norm(v, dim=-1, keepdim=True)

    phi = torch.atan2(vnorm, w.clamp(-1.0, 1.0))

    u = torch.where(
        vnorm > 1e-8,
        v / vnorm,
        torch.zeros_like(v)
    )

    return u * phi


def q_rotate(q, vec):
    """
    v' = q^{-1} [0,v] q
    """
    q = q_normalize(q)

    vq = torch.cat([
        torch.zeros_like(vec[..., :1]),
        vec
    ], dim=-1)

    return q_mul(q_mul(q_inv(q), vq), q)[..., 1:]
