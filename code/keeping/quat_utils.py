# quat_utils.py
import torch

# Quaternion convention: q = [w, x, y, z]
# Pure vector v in R^3 is represented as [0, vx, vy, vz]

def q_normalize(q, eps=1e-12):
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)

def q_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def q_inv(q, eps=1e-12):
    # For unit quaternion, inverse = conjugate. Keep generic:
    return q_conj(q) / (torch.sum(q*q, dim=-1, keepdim=True) + eps)

def q_mul(q1, q2):
    # Hamilton product
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def q_exp_pure(v):
    """
    exp([0, v]) where v in R^3 (axis-angle/2 style vector)
    Returns unit quaternion.
    """
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)  # (...,1)
    half = theta
    # when theta is small, sin(theta)/theta ~ 1
    sinc = torch.where(theta > 1e-8, torch.sin(half)/theta, torch.ones_like(theta))
    w = torch.cos(half)
    xyz = sinc * v
    return torch.cat([w, xyz], dim=-1)

def q_log(q):
    """
    log of a unit quaternion.
    Returns axis-angle vector (in R^3) such that exp([0, v]) = q.
    """
    q = q_normalize(q)
    w = q[..., :1]
    v = q[..., 1:]
    vnorm = torch.linalg.norm(v, dim=-1, keepdim=True)
    # angle = arctan2(|v|, w) * 2, but here we return v_axis * angle/2 style?
    # For consistency with cost in PDF: they use 2*log(q_rel) then norm^2.
    # We'll return u*phi where phi = arctan2(|v|, w), u = v/|v|. (this is "log" in quaternion algebra)
    phi = torch.atan2(vnorm, w.clamp(min=-1.0, max=1.0))
    u = torch.where(vnorm > 1e-8, v / vnorm, torch.zeros_like(v))
    return u * phi  # (...,3)

def q_rotate(q, vec):
    """
    Rotate 3D vector vec by quaternion q: v' = q^{-1} [0,v] q  (IMU convention in PDF obs model)
    vec: (...,3)
    """
    q = q_normalize(q)
    vq = torch.cat([torch.zeros_like(vec[..., :1]), vec], dim=-1)
    return q_mul(q_mul(q_inv(q), vq), q)[..., 1:]