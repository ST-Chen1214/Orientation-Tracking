# imu_calib.py (clean)
import numpy as np


def calibrate_imu(imud,
                  static_seconds=3.0,
                  acc_sensitivity_mv_per_g=None,
                  gyro_sensitivity_mv_per_rad_s=None,
                  vref_mv=3300.0,
                  adc_bits=10):
    """
    Input:
      imud: ndarray (7,N) or (N,7)
            [ts ax ay az gx gy gz]

    Return:
      ts (N,)
      omega (N,3)   rad/s
      acc_g (N,3)   in g
    """

    # -----------------------
    # 1. Extract ts / vals
    # -----------------------
    if not isinstance(imud, np.ndarray):
        raise ValueError("imud must be numpy array")

    if imud.shape[0] == 7:        # (7,N)
        ts = imud[0]
        vals = imud[1:7]

    elif imud.shape[1] == 7:      # (N,7)
        ts = imud[:, 0]
        vals = imud[:, 1:7].T

    else:
        raise ValueError(f"Bad IMU shape: {imud.shape}")

    ts = ts.astype(np.float64)
    vals = vals.astype(np.float64)   # (6,N)


    # -----------------------
    # 2. ADC -> mV
    # -----------------------
    mv_per_count = vref_mv / (2**adc_bits - 1)
    vals_mv = vals * mv_per_count

    acc_mv = vals_mv[0:3].T     # (N,3)
    gyro_mv = vals_mv[3:6].T    # (N,3)


    # -----------------------
    # 3. Unit convert
    # -----------------------
    if acc_sensitivity_mv_per_g is None or gyro_sensitivity_mv_per_rad_s is None:
        raise ValueError("Please set sensor sensitivities")

    acc_g = acc_mv / acc_sensitivity_mv_per_g
    omega = gyro_mv / gyro_sensitivity_mv_per_rad_s


    # -----------------------
    # 4. Bias removal
    # -----------------------
    t0 = ts[0]
    static_mask = (ts - t0) <= static_seconds

    if static_mask.sum() < 10:
        static_mask = np.arange(min(200, len(ts)))

    omega_bias = omega[static_mask].mean(axis=0)
    acc_bias = acc_g[static_mask].mean(axis=0) - np.array([0, 0, 1])

    omega -= omega_bias
    acc_g -= acc_bias


    return ts, omega, acc_g