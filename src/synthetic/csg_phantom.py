"""Voxelised CSG phantoms with thin ribs, rounded joints, and open channels."""
import numpy as np


def _rounded_box_sdf(x, y, z, centre, half_extent, radius):
    qx = np.abs(x - centre[0]) - (half_extent[0] - radius)
    qy = np.abs(y - centre[1]) - (half_extent[1] - radius)
    qz = np.abs(z - centre[2]) - (half_extent[2] - radius)
    outside = np.sqrt(
        np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2
    )
    return outside + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0) - radius


def _ellipsoid_sdf(x, y, z, axes):
    return (np.sqrt((x / axes[0]) ** 2 + (y / axes[1]) ** 2 +
                    (z / axes[2]) ** 2) - 1.0) * min(axes)


def _smooth_union(a, b, radius):
    h = np.maximum(radius - np.abs(a - b), 0.0) / radius
    return np.minimum(a, b) - h * h * radius * 0.25


def csg_density(n, params):
    """Return a float32 occupancy volume in [-1, 1]^3 for one deterministic CSG."""
    if n < 8:
        raise ValueError("grid resolution must be at least 8")
    scale = float(params.get("body_scale", 1.0))
    rib = float(params.get("rib_thickness", 0.034))
    channel_radius = float(params.get("channel_radius", 0.04))
    if scale <= 0 or rib <= 0 or channel_radius <= 0:
        raise ValueError("CSG dimensions must be positive")

    axis = (-1.0 + (np.arange(n, dtype=np.float32) + 0.5) * (2.0 / n))
    x, y = axis[:, None], axis[None, :]
    volume = np.zeros((n, n, n), dtype=np.float32)
    body_axes = np.array([0.43, 0.38, 0.34], dtype=np.float32) * scale

    for iz, z0 in enumerate(axis):
        z = np.full_like(x, z0)
        body = _ellipsoid_sdf(x, y, z, body_axes)

        # Ribs overlap the body, while rounded box corners act as filleted joints.
        rib_x = _rounded_box_sdf(
            x, y, z, (0.34 * scale, 0.0, 0.16 * scale),
            (0.34 * scale, rib * 0.5, rib * 1.6), rib * 0.35,
        )
        rib_y = _rounded_box_sdf(
            x, y, z, (-0.12 * scale, -0.29 * scale, -0.12 * scale),
            (rib * 0.5, 0.28 * scale, rib * 1.5), rib * 0.35,
        )
        bridge = _rounded_box_sdf(
            x, y, z, (0.19 * scale, -0.17 * scale, 0.02 * scale),
            (0.17 * scale, rib * 0.75, rib * 1.2), rib * 0.45,
        )
        solid = _smooth_union(body, rib_x, rib * 0.8)
        solid = _smooth_union(solid, rib_y, rib * 0.8)
        solid = _smooth_union(solid, bridge, rib * 0.8)

        # A channel deliberately opens at both z ends so segmentation hole-filling
        # does not erase the feature.
        radial = np.sqrt(x * x + y * y) - channel_radius
        axial = np.abs(z) - 0.62 * scale
        channel = np.minimum(np.maximum(radial, axial), 0.0) + np.sqrt(
            np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2
        )
        volume[:, :, iz] = (np.maximum(solid, -channel) <= 0).astype(np.float32)
    return volume
