# panorama.py
import numpy as np
import transforms3d.quaternions as t3q


def _unwrap_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _unwrap_singleton(x[0])
    if isinstance(x, np.ndarray) and x.size == 1:
        return _unwrap_singleton(x.flat[0])
    return x


def _to_hwc_uint8(img):
    img = np.asarray(img)

    # grayscale
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    # (H,3,W) -> (H,W,3)
    if img.ndim == 3 and img.shape[1] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (0, 2, 1))

    # (3,H,W) -> (H,W,3)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape after conversion: {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def extract_cam_ts_imgs(camd):
    # dict
    if isinstance(camd, dict):
        cam_ts = _unwrap_singleton(camd.get("ts"))
        imgs = _unwrap_singleton(camd.get("cam"))
        if cam_ts is None or imgs is None:
            raise ValueError(f"cam dict keys={list(camd.keys())}, expected 'ts' and 'cam'")
        cam_ts = np.array(cam_ts, dtype=np.float64).squeeze()
        return cam_ts, imgs

    # structured ndarray
    if isinstance(camd, np.ndarray) and camd.dtype.names is not None:
        names = camd.dtype.names
        if "ts" in names and "cam" in names:
            cam_ts = np.array(_unwrap_singleton(camd["ts"]), dtype=np.float64).squeeze()
            imgs = _unwrap_singleton(camd["cam"])
            return cam_ts, imgs
        raise ValueError(f"cam structured fields={names}, expected 'ts' and 'cam'")

    # plain ndarray (assume first axis is frame)
    if isinstance(camd, np.ndarray) and camd.ndim >= 3:
        M = camd.shape[0]
        cam_ts = np.arange(M, dtype=np.float64)
        return cam_ts, camd

    raise ValueError(f"Unknown cam format: type={type(camd)}, dtype={getattr(camd,'dtype',None)}, shape={getattr(camd,'shape',None)}")


def _num_frames(imgs):
    """Return number of frames in imgs, whether list-like or ndarray."""
    if isinstance(imgs, (list, tuple)):
        return len(imgs)
    if isinstance(imgs, np.ndarray):
        return imgs.shape[0]
    raise ValueError(f"Unsupported imgs container type: {type(imgs)}")


def _get_frame(imgs, i):
    """Fetch i-th frame from imgs."""
    if isinstance(imgs, (list, tuple)):
        return imgs[i]
    return imgs[i]


def build_panorama(
    camd, ts_imu, q_imu,
    pano_h=400, pano_w=1200,
    fx=250, fy=250,
    cx=None, cy=None,
    use_Rt=True
):
    cam_ts, imgs = extract_cam_ts_imgs(camd)

    # ✅ Fix mismatch between timestamps and frames
    M_img = _num_frames(imgs)
    M_ts = int(np.array(cam_ts).shape[0])
    M = min(M_img, M_ts)

    if M < 1:
        raise ValueError(f"No camera frames found. M_img={M_img}, M_ts={M_ts}")

    if M_ts != M_img:
        print(f"[WARN] cam_ts length ({M_ts}) != num images ({M_img}). Using M={M} frames.")

    # Determine H,W from first frame (converted)
    first = _to_hwc_uint8(_get_frame(imgs, 0))
    H, W, _ = first.shape

    if cx is None: cx = (W - 1) / 2.0
    if cy is None: cy = (H - 1) / 2.0

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    rays = np.stack([x, y, z], axis=-1)
    rays /= (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-12)

    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

    def imu_index_for_cam(t):
        idx = np.searchsorted(ts_imu, t, side="right") - 1
        return int(np.clip(idx, 0, len(ts_imu) - 1))

    # ✅ iterate only valid frames
    for i in range(M):
        t = cam_ts[i]
        idx = imu_index_for_cam(t)
        q = q_imu[idx]
        R = t3q.quat2mat(q)

        rw = rays @ (R.T if use_Rt else R) # body to world

        az = np.arctan2(rw[..., 1], rw[..., 0])
        el = np.arcsin(np.clip(rw[..., 2], -1.0, 1.0))

        # px = ((az + np.pi) / (2 * np.pi) * (pano_w - 1)).astype(np.int32)
        # py = ((np.pi / 2 - el) / np.pi * (pano_h - 1)).astype(np.int32)
        px = ((az + np.pi) / (2 * np.pi) * (pano_w - 1)).astype(np.int32)
        py = ((el - np.pi / 2) / np.pi * (pano_h - 1)).astype(np.int32)

        img = _to_hwc_uint8(_get_frame(imgs, i))
        pano[py, px] = img

    return pano
