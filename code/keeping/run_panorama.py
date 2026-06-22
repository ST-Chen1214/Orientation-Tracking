# run_panorama.py
import os
import numpy as np
import matplotlib.pyplot as plt

from load_data import read_data
from imu_calib import calibrate_imu
from panorama import build_panorama
from orient_pgd import pgd_optimize   # 直接沿用你已經跑通的

# -----------------------------
# Data loading
# -----------------------------
def load_seq(dataset_id="8", base="../data/trainset"):
    ifile = f"{base}/imu/imuRaw{dataset_id}.p"
    cfile = f"{base}/cam/cam{dataset_id}.p"
    imud = read_data(ifile)
    camd = read_data(cfile)
    return camd, imud


def main():
    dataset_id = "2"  # 改成有相機的那組（通常 4 組有 cam）, 只有cam有用
    camd, imud = load_seq(dataset_id)

    # TODO: 填你的 sensitivity（跟你姿態那份一樣）
    ACC_SENS = 330.0
    GYRO_SENS = 190.8

    ts, omega, acc_g = calibrate_imu(
        imud,
        static_seconds=3.0,
        acc_sensitivity_mv_per_g=ACC_SENS,
        gyro_sensitivity_mv_per_rad_s=GYRO_SENS
    )

    q_pgd = pgd_optimize(ts, omega, acc_g, iters=200, lr=1e-2)

    pano = build_panorama(camd, ts, q_pgd, pano_h=400, pano_w=1200)

    out = "panorama.png"
    plt.figure(figsize=(12,4))
    plt.imshow(pano)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print("Saved:", os.path.abspath(out))

if __name__ == "__main__":
    main()
