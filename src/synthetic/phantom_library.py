"""Generates paired clean-reference and degraded-input phantom instances."""
import json
import os
import shutil

import numpy as np
import scipy.ndimage
import trimesh
import yaml
from scipy.ndimage import gaussian_filter
from skimage import measure

from src.synthetic.phantom_params import SHEPP_LOGAN_3D
from src.synthetic.csg_phantom import csg_density

HEAD_LEVEL = 0.5    # air is 0, tissue is >= 1.0, so 0.5 separates head from air


def load_config(config_path="configs/synthetic.yaml"):
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


class PhantomLibrary:
    def __init__(self, config_or_path="configs/synthetic.yaml", out_dir=None, n_clean=None, n_deg=None):
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config = load_config(config_or_path)

        grid_cfg = self.config.get("grid", {})
        self.n_clean = n_clean if n_clean is not None else grid_cfg.get("n_clean", 256)
        self.n_deg = n_deg if n_deg is not None else grid_cfg.get("n_deg", 128)
        self.out_dir = out_dir if out_dir is not None else self.config.get("out_dir", "data/synthetic")

        self.geom_cfg = self.config.get("geometry", {
            "axis_jitter": 0.05,
            "centre_jitter": 0.01,
            "rotation_jitter_deg": 2.0,
            "max_extent": 0.95,
        })
        self.geometry_kind = self.geom_cfg.get("kind", "csg")
        self.tiers = self.config.get("tiers", {
            "clean":    dict(blur=0.0, noise=0.00, seg_threshold=0.5),
            "moderate": dict(blur=1.0, noise=0.05, seg_threshold=0.5),
            "poor":     dict(blur=2.0, noise=0.10, seg_threshold=0.7),
        })

        assert self.n_clean % self.n_deg == 0, "clean grid must be an integer multiple of the degraded grid"
        self.f = self.n_clean // self.n_deg
        self.dx_clean = 2.0 / self.n_clean
        self.dx_deg = 2.0 / self.n_deg
        self.g_clean = self._axis(self.n_clean)

        # Compute nominal unjittered head voxel count once at startup
        if self.geometry_kind == "csg":
            nominal_rib = 2.25 * 2.0 / self.n_deg
            nom_dens = csg_density(
                self.n_clean,
                {"rib_thickness": nominal_rib, "channel_radius": 0.0425},
            )
        elif self.geometry_kind == "shepp_logan":
            nominal_params = {
                k: (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
                for k, v in SHEPP_LOGAN_3D.items()
            }
            nom_dens = self.density(nominal_params)
        else:
            raise ValueError(f"unknown phantom geometry kind: {self.geometry_kind}")
        self.nominal_voxels = int(np.sum(nom_dens >= HEAD_LEVEL))

        for sub in ("specs", "reference", "density", "degraded", "logs"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

    @staticmethod
    def _axis(n):
        return (-1 + (np.arange(n) + 0.5) * (2.0 / n)).astype(np.float32)

    @staticmethod
    def _ellipsoid_mask(gx, gy, gz, phi, a, b, c, x0, y0, z0):
        x, y, z = gx - x0, gy - y0, gz - z0
        cp, sp = np.cos(phi), np.sin(phi)
        xr = (x * cp + y * sp) / a**2
        yr = (-x * sp + y * cp) / b**2
        zr = z / c**2
        return (xr**2 + yr**2 + zr**2) <= 1.0

    def sample_params(self, rng):
        if self.geometry_kind == "csg":
            return {
                "seed": int(rng.integers(0, 2**31)),
                "body_scale": float(rng.uniform(0.96, 1.04)),
                "rib_thickness": float(rng.uniform(1.5, 3.0) * 2.0 / self.n_deg),
                "channel_radius": float(rng.uniform(0.035, 0.05)),
            }
        axis_j = self.geom_cfg.get("axis_jitter", 0.05)
        centre_j = self.geom_cfg.get("centre_jitter", 0.01)
        rot_j = self.geom_cfg.get("rotation_jitter_deg", 2.0)
        axis_scale = rng.uniform(1.0 - axis_j, 1.0 + axis_j, 3)
        centre_shift = rng.normal(0, centre_j, 3)
        rotation_shift = rng.uniform(-rot_j, rot_j)
        params = {}
        for name, (delta, phi, a, b, c, x0, y0, z0) in SHEPP_LOGAN_3D.items():
            axes = np.array([a, b, c]) * axis_scale
            centre = np.array([x0, y0, z0]) + centre_shift
            params[name] = (delta, phi + rotation_shift, *axes.tolist(), *centre.tolist())
        return params

    def density(self, params):
        if self.geometry_kind == "csg":
            return csg_density(self.n_clean, params)
        g = self.g_clean
        gx, gy, gz = g[:, None, None], g[None, :, None], g[None, None, :]
        d = np.zeros((self.n_clean,) * 3, dtype=np.float32)
        for delta, phi_deg, a, b, c, x0, y0, z0 in params.values():
            m = self._ellipsoid_mask(gx, gy, gz, np.deg2rad(phi_deg), a, b, c, x0, y0, z0)
            d[m] += delta
        return d

    def check_geometry(self, dens):
        min_val = float(dens.min())
        border = np.concatenate([
            dens[0].ravel(), dens[-1].ravel(),
            dens[:, 0].ravel(), dens[:, -1].ravel(),
            dens[:, :, 0].ravel(), dens[:, :, -1].ravel()
        ])
        border_max = float(border.max())
        head_mask = dens >= HEAD_LEVEL
        head_voxels = int(np.sum(head_mask))
        ratio = head_voxels / max(self.nominal_voxels, 1)

        _, n_components = scipy.ndimage.label(head_mask)

        metrics = {
            "min_density": min_val,
            "border_max": border_max,
            "head_voxels": head_voxels,
            "nominal_voxels": self.nominal_voxels,
            "volume_ratio": float(ratio),
            "n_components": int(n_components),
        }

        if min_val < -1e-6:
            return False, "negative_density", metrics
        if border_max >= HEAD_LEVEL:
            return False, "touches_border", metrics
        if ratio < 0.6 or ratio > 1.4:
            return False, "volume_out_of_range", metrics
        if n_components != 1:
            return False, "not_single_component", metrics

        return True, "ok", metrics

    def degrade(self, dens, tier, rng):
        t = self.tiers[tier]
        if t["blur"] > 0:
            dens = gaussian_filter(dens, sigma=t["blur"] * self.f)
        n, f = self.n_deg, self.f
        vol = dens.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5))
        if t["noise"] > 0:
            vol = vol + rng.normal(0.0, t["noise"], vol.shape)
        return vol.astype(np.float32)

    def _reference_mesh(self, dens):
        verts, faces, _, _ = measure.marching_cubes(dens, level=HEAD_LEVEL,
                                                    spacing=(self.dx_clean,) * 3)
        verts = verts + (-1.0 + 0.5 * self.dx_clean)
        return trimesh.Trimesh(verts, faces)

    def write_instance_atomic(self, tag, seed, params, dens, noise_rng):
        tmp_dir = os.path.join(self.out_dir, ".tmp", tag)
        for sub in ("specs", "reference", "density", "degraded"):
            os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)

        try:
            # 1. Reference mesh
            ref = self._reference_mesh(dens)
            tmp_ref_path = os.path.join(tmp_dir, "reference", f"{tag}_ref.stl")
            ref.export(tmp_ref_path)

            # 2. Clean density
            tmp_density_path = os.path.join(tmp_dir, "density", f"{tag}.npz")
            np.savez_compressed(tmp_density_path, density=dens.astype(np.float16))

            # 3. Specs json
            tmp_spec_path = os.path.join(tmp_dir, "specs", f"{tag}.json")
            with open(tmp_spec_path, "w") as fh:
                serial_params = {
                    k: ([float(x) for x in v] if isinstance(v, (tuple, list, np.ndarray))
                        else float(v))
                    for k, v in params.items()
                }
                json.dump({"id": tag, "seed": seed, "params": serial_params},
                          fh, indent=1)

            # 4. Degraded tiers in clean, moderate, poor order
            rows = []
            final_ref_path = os.path.join(self.out_dir, "reference", f"{tag}_ref.stl")
            final_density_path = os.path.join(self.out_dir, "density", f"{tag}.npz")
            final_spec_path = os.path.join(self.out_dir, "specs", f"{tag}.json")

            for tier_name in ("clean", "moderate", "poor"):
                t_cfg = self.tiers[tier_name]
                vol = self.degrade(dens, tier_name, noise_rng)
                tmp_vol_path = os.path.join(tmp_dir, "degraded", f"{tag}_{tier_name}.npy")
                np.save(tmp_vol_path, vol.astype(np.float16))

                final_vol_path = os.path.join(self.out_dir, "degraded", f"{tag}_{tier_name}.npy")
                rows.append({
                    "id": tag,
                    "tier": tier_name,
                    "seed": seed,
                    "volume_path": final_vol_path,
                    "seg_threshold": t_cfg["seg_threshold"],
                    "reference_path": final_ref_path,
                    "density_path": final_density_path,
                    "spec_path": final_spec_path,
                    "ref_watertight": bool(ref.is_watertight),
                })

            # Move files atomically to final locations
            for sub, filename in [
                ("reference", f"{tag}_ref.stl"),
                ("density", f"{tag}.npz"),
                ("specs", f"{tag}.json"),
            ]:
                src = os.path.join(tmp_dir, sub, filename)
                dst = os.path.join(self.out_dir, sub, filename)
                shutil.move(src, dst)

            for tier_name in ("clean", "moderate", "poor"):
                fn = f"{tag}_{tier_name}.npy"
                src = os.path.join(tmp_dir, "degraded", fn)
                dst = os.path.join(self.out_dir, "degraded", fn)
                shutil.move(src, dst)

            shutil.rmtree(tmp_dir, ignore_errors=True)
            return rows

        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
