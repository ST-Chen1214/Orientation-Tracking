# panorama.py (clean)
import numpy as np
import transforms3d.quaternions as t3q


# -----------------------------
# Camera data
# -----------------------------
def extract_cam(camd):
    """
    Return:
      cam_ts (M,)
      imgs   (M,H,W,3) or list
    """

    # dict (standard dataset)
    if isinstance(camd, dict):
        return np.asarray(camd["ts"]).squeeze(), camd["cam"]

    # ndarray (fallback)
    if isinstance(camd, np.ndarray):
        M = camd.shape[0]
        return np.arange(M, dtype=np.float64), camd



def to_uint8(img):

    img = np.asarray(img)

    # grayscale → RGB
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    # (3,H,W) → (H,W,3)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # (H,3,W) → (H,W,3)
    if img.shape[1] == 3:
        img = np.transpose(img, (0, 2, 1))

    return np.clip(img, 0, 255).astype(np.uint8)


# -----------------------------
# Panorama
# -----------------------------
def build_panorama(
    camd,
    ts_imu,
    q_imu,
    pano_h=400,
    pano_w=1200,
    fx=250,
    fy=250,
    cx=None,
    cy=None,
    use_Rt=True
):

    cam_ts, imgs = extract_cam(camd)

    M = min(len(cam_ts), len(imgs))
    if M == 0:
        raise ValueError("No camera frames")

    # image size
    img0 = to_uint8(imgs[0])
    H, W, _ = img0.shape

    if cx is None:
        cx = (W - 1) / 2
    if cy is None:
        cy = (H - 1) / 2


    # -----------------------------
    # Ray directions (camera frame)
    # -----------------------------
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)

    rays = np.stack([x, y, z], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)


    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)


    # IMU timestamp match
    def imu_idx(t):
        i = np.searchsorted(ts_imu, t, side="right") - 1
        return int(np.clip(i, 0, len(ts_imu) - 1))


    # -----------------------------
    # Main loop
    # -----------------------------
    for i in range(M):

        idx = imu_idx(cam_ts[i])
        q = q_imu[idx]

        R = t3q.quat2mat(q)

        # rotate rays
        if use_Rt:
            rw = rays @ R.T
        else:
            rw = rays @ R


        # spherical projection
        az = np.arctan2(rw[..., 1], rw[..., 0])
        el = np.arcsin(np.clip(rw[..., 2], -1, 1))


        px = ((az + np.pi) / (2*np.pi) * (pano_w - 1)).astype(int)
        py = ((el - np.pi/2) / np.pi * (pano_h - 1)).astype(int)


        img = to_uint8(imgs[i])

        pano[py, px] = img


    return pano
