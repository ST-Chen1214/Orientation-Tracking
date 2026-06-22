# run_panorama.py (clean)
import numpy as np
import matplotlib.pyplot as plt
import os

from load_data import read_data
from imu_calib import calibrate_imu
from panorama import build_panorama
from orient_pgd import pgd_optimize


# -----------------------------
# Data loading
# -----------------------------
def load_seq(dataset_id, base="../data/trainset"):

    imud = read_data(f"{base}/imu/imuRaw{dataset_id}.p")
    camd = read_data(f"{base}/cam/cam{dataset_id}.p")

    return camd, imud


# -----------------------------
# Main
# -----------------------------
def main():

    DATASET = "11"      # must have camera
    ACC = 330.0
    GYRO = 190.8

    camd, imud = load_seq(DATASET)

    # IMU → physical units
    ts, omega, acc = calibrate_imu(
        imud,
        acc_sensitivity_mv_per_g=ACC,
        gyro_sensitivity_mv_per_rad_s=GYRO
    )

    # Same PGD as orient_pgd.py
    q = pgd_optimize(ts, omega, acc)

    # Panorama
    pano = build_panorama(
        camd, ts, q,
        pano_h=400,
        pano_w=1200
    )

    # Save / show
    plt.figure(figsize=(12, 4))
    plt.imshow(pano)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("panorama.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
