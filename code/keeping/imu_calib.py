# imu_calib.py
import numpy as np

def _extract_ts_vals(imud):
    import numpy as np

    # Case: plain numeric ndarray (your current case: dtype=float64)
    if isinstance(imud, np.ndarray) and imud.dtype != object and imud.dtype.names is None:
        if imud.ndim != 2:
            raise ValueError(f"Plain ndarray must be 2D. Got shape={imud.shape}")

        r, c = imud.shape

        # (7, N): [ts; ax ay az gx gy gz]
        if r == 7:
            ts = imud[0, :]
            vals = imud[1:7, :]   # (6, N)

        # (N, 7): [ts ax ay az gx gy gz]
        elif c == 7:
            ts = imud[:, 0]
            vals = imud[:, 1:7].T  # (6, N)

        # (6, N): only vals, no ts (fallback: fake ts with constant dt)
        elif r == 6:
            vals = imud
            N = vals.shape[1]
            ts = np.arange(N, dtype=np.float64) * 0.01  # fallback dt=0.01s

        # (N, 6): only vals, no ts
        elif c == 6:
            vals = imud.T
            N = vals.shape[1]
            ts = np.arange(N, dtype=np.float64) * 0.01

        else:
            raise ValueError(f"Cannot infer ts/vals from ndarray shape={imud.shape}. "
                             f"Expected (7,N) or (N,7) or (6,N)/(N,6).")

        ts = np.array(ts).astype(np.float64).squeeze()
        vals = np.array(vals).astype(np.float64)

        # ensure (6,N)
        if vals.shape[0] != 6 and vals.shape[1] == 6:
            vals = vals.T
        if vals.shape[0] != 6:
            raise ValueError(f"vals must be (6,N). Got {vals.shape}")

        return ts, vals

    # dict
    if isinstance(imud, dict) and 'ts' in imud and 'vals' in imud:
        ts = imud['ts']
        vals = imud['vals']
        if isinstance(ts, (list, tuple)) and len(ts) == 1:
            ts = ts[0]
        if isinstance(vals, (list, tuple)) and len(vals) == 1:
            vals = vals[0]
        ts = np.array(ts).astype(np.float64).squeeze()
        vals = np.array(vals).astype(np.float64)
        if vals.shape[0] != 6 and vals.shape[1] == 6:
            vals = vals.T
        if vals.shape[0] != 6:
            raise ValueError(f"vals must be (6,N). Got {vals.shape}")
        return ts, vals

    # structured ndarray
    if isinstance(imud, np.ndarray) and imud.dtype.names is not None:
        ts = imud['ts']
        vals = imud['vals']
        # peel singletons
        while isinstance(ts, np.ndarray) and ts.size == 1:
            ts = ts.flat[0]
        while isinstance(vals, np.ndarray) and vals.size == 1 and vals.dtype == object:
            vals = vals.flat[0]
        ts = np.array(ts).astype(np.float64).squeeze()
        vals = np.array(vals).astype(np.float64)
        if vals.shape[0] != 6 and vals.shape[1] == 6:
            vals = vals.T
        if vals.shape[0] != 6:
            raise ValueError(f"vals must be (6,N). Got {vals.shape}")
        return ts, vals

    # object ndarray
    if isinstance(imud, np.ndarray) and imud.dtype == object and imud.size == 2:
        ts = np.array(imud.flat[0]).astype(np.float64).squeeze()
        vals = np.array(imud.flat[1]).astype(np.float64)
        if vals.shape[0] != 6 and vals.shape[1] == 6:
            vals = vals.T
        if vals.shape[0] != 6:
            raise ValueError(f"vals must be (6,N). Got {vals.shape}")
        return ts, vals

    raise ValueError(f"Unknown imud format: type={type(imud)}, dtype={getattr(imud,'dtype',None)}, shape={getattr(imud,'shape',None)}")




def calibrate_imu(imud, static_seconds=3.0,
                  acc_sensitivity_mv_per_g=None,
                  gyro_sensitivity_mv_per_rad_s=None,
                  vref_mv=3300.0, adc_bits=10):
    """
    returns:
      ts (N,)
      omega_cal (N,3) rad/s
      acc_cal (N,3) in gravity units (g)
    """
    # ✅ 用 robust extractor 取代你原本的 imud['ts'][0] / imud['vals'][0]
    ts, vals = _extract_ts_vals(imud)   # vals: (6, N)

    # ADC counts -> mV
    mv_per_count = vref_mv / (2**adc_bits - 1)
    vals_mv = vals * mv_per_count

    acc_mv = vals_mv[0:3, :].T   # (N,3)
    gyro_mv = vals_mv[3:6, :].T  # (N,3)

    if acc_sensitivity_mv_per_g is None or gyro_sensitivity_mv_per_rad_s is None:
        raise ValueError("請先填入 datasheet 的 sensitivity：acc_sensitivity_mv_per_g, gyro_sensitivity_mv_per_rad_s")

    # Convert to physical units (before bias removal)
    acc_g = acc_mv / acc_sensitivity_mv_per_g
    omega = gyro_mv / gyro_sensitivity_mv_per_rad_s

    # static segment
    t0 = ts[0]
    static_mask = (ts - t0) <= static_seconds
    if static_mask.sum() < 10:
        static_mask = np.arange(min(200, len(ts)))

    omega_bias = omega[static_mask].mean(axis=0)
    acc_bias = acc_g[static_mask].mean(axis=0) - np.array([0.0, 0.0, 1.0])

    omega_cal = omega - omega_bias
    acc_cal = acc_g - acc_bias

    return ts, omega_cal, acc_cal