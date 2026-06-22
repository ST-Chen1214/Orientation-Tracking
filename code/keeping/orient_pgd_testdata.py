# orient_pgd.py (IMU only)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from load_data import read_data
from imu_calib import calibrate_imu
from quat_utils import q_mul, q_inv, q_exp_pure, q_log, q_rotate, q_normalize


# -----------------------------
# Data loading (IMU only)
# -----------------------------
def load_seq(dataset_id="1", base="../data/trainset"):
    ifile = f"{base}/imu/imuRaw{dataset_id}.p"
    imud = read_data(ifile)
    return imud


# -----------------------------
# Utilities
# -----------------------------
def integrate_gyro(ts, omega):
    """
    Simple quaternion integration:
      q_{k+1} = q_k ∘ exp([0, dt*omega/2])
    Returns q (N,4) in [w,x,y,z]
    """
    N = len(ts)
    q = np.zeros((N, 4), dtype=np.float64)
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(N - 1):
        dt = ts[k + 1] - ts[k]
        v = (dt * omega[k]) / 2.0  # (3,)
        theta = np.linalg.norm(v)
        if theta < 1e-8:
            dq = np.array([1.0, v[0], v[1], v[2]])
        else:
            dq = np.hstack([np.cos(theta), np.sin(theta) * v / theta])

        # Hamilton product q[k] ∘ dq
        w1, x1, y1, z1 = q[k]
        w2, x2, y2, z2 = dq
        q[k + 1] = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
        q[k + 1] /= (np.linalg.norm(q[k + 1]) + 1e-12)
    return q


def q_to_rpy_deg(qs):
    """
    qs: (N,4) [w,x,y,z]
    Return: (N,3) [roll, pitch, yaw] degrees, ZYX (yaw-pitch-roll)
    """
    qs = np.asarray(qs, dtype=np.float64)
    w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]

    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    yaw = np.arctan2(r10, r00)
    pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
    roll = np.arctan2(r21, r22)

    rpy = np.stack([roll, pitch, yaw], axis=1) * 180.0 / np.pi
    return rpy


# -----------------------------
# PGD optimizer (IMU only)
# -----------------------------
def pgd_optimize(ts, omega, acc_g, iters=200, lr=1e-2, device="cpu"):
    ts_t = torch.tensor(ts, dtype=torch.float32, device=device)
    omega_t = torch.tensor(omega, dtype=torch.float32, device=device)
    acc_t = torch.tensor(acc_g, dtype=torch.float32, device=device)

    q0_np = integrate_gyro(ts, omega)
    q = torch.tensor(q0_np, dtype=torch.float32, device=device, requires_grad=True)

    g_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    for it in range(iters):
        dt = (ts_t[1:] - ts_t[:-1]).unsqueeze(-1)          # (N-1,1)
        v = (dt * omega_t[:-1]) / 2.0                      # (N-1,3)
        dq = q_exp_pure(v)                                 # (N-1,4)
        q_pred = q_mul(q[:-1], dq)                         # (N-1,4)

        q_rel = q_mul(q_inv(q[1:]), q_pred)
        log_rel = q_log(q_rel)
        motion_err = 2.0 * log_rel
        c1 = 0.5 * torch.sum(motion_err * motion_err)

        a_pred = q_rotate(q, g_world.expand_as(acc_t))
        obs_err = acc_t - a_pred
        c2 = 0.5 * torch.sum(obs_err * obs_err)

        cost = c1 + c2
        cost.backward()

        with torch.no_grad():
            q -= lr * q.grad
            q[:] = q_normalize(q)
            q.grad.zero_()

        if (it + 1) % 20 == 0:
            print(
                f"iter {it+1:4d} cost={cost.item():.4e}  "
                f"motion={c1.item():.4e}  obs={c2.item():.4e}"
            )

    return q.detach().cpu().numpy()


# -----------------------------
# Main
# -----------------------------
def main():
    dataset_id = "11"  # IMU only：任何有 imuRaw*.p 的都可
    base = "../data/trainset"
    imud = load_seq(dataset_id, base=base)

    # Debug: confirm which imu_calib is being used
    import imu_calib as _ic
    print("USING imu_calib:", os.path.abspath(_ic.__file__))
    print("imud ndarray shape:", getattr(imud, "shape", None))

    # TODO: 請填正確 sensitivity（mV/g 與 mV/(rad/s)）
    ACC_SENS = 330.0    # placeholder
    GYRO_SENS = 190.8   # placeholder

    ts, omega, acc_g = calibrate_imu(
        imud,
        static_seconds=3.0,
        acc_sensitivity_mv_per_g=ACC_SENS,
        gyro_sensitivity_mv_per_rad_s=GYRO_SENS
    )

    # Baseline integration (IMU gyro only)
    q_int = integrate_gyro(ts, omega)

    # PGD (gyro + accel gravity constraint)
    q_pgd = pgd_optimize(ts, omega, acc_g, iters=200, lr=1e-2)

    # Plot: PGD vs Gyro Integration
    rpy_pgd = q_to_rpy_deg(q_pgd)
    rpy_int = q_to_rpy_deg(q_int)

    fig = plt.figure(figsize=(10, 6))
    names = ["Roll", "Pitch", "Yaw"]
    for k in range(3):
        ax = fig.add_subplot(3, 1, k + 1)
        # ax.plot(rpy_int[:, k], label="GyroInt (IMU)")
        ax.plot(rpy_pgd[:, k], label="PGD")
        if k == 0:
            ax.legend()
        ax.set_ylabel(names[k] + " (deg)")
    ax.set_xlabel("IMU index")
    plt.tight_layout()

    out_png = "rpy_plot_imu_only.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")

    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed (ok). Error:", e)


if __name__ == "__main__":
    main()
