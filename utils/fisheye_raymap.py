import numpy as np

def fisheye_raymap(H, W):
    cx = W / 2.0
    cy = H / 2.0
    R = min(H, W) / 2.0

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    x = (xx - cx) / R
    y = (yy - cy) / R

    r = np.sqrt(x**2 + y**2)

    # 180° fisheye: image circle edge r=1 corresponds to theta=90°
    theta = r * (np.pi / 2.0)
    phi = np.arctan2(y, x)

    ray_x = np.sin(theta) * np.cos(phi)
    ray_y = np.sin(theta) * np.sin(phi)
    ray_z = np.cos(theta)

    ray = np.stack([ray_x, ray_y, ray_z], axis=0).astype(np.float32)  # [3,H,W]

    # outside fisheye circle: mark invalid
    valid = (r <= 1.0).astype(np.float32)[None]  # [1,H,W]
    ray = ray * valid

    return ray, valid