from pathlib import Path
import filecmp
import h5py
from h5py import File as h5File
import numpy as np
import shutil
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

source_dir: Path = Path("/anvil/projects/x-ees240016/TurbChannelInit/")
dest_dir: Path = Path("/anvil/scratch/x-lwidmer/RUN13")

# source_dir: Path = Path("~/Documents/UCSB/TurbChannelInit/").expanduser()
# dest_dir: Path = Path("~/Documents/UCSB/ANVIL_scripts/tmp/").expanduser()
# dest_dir.mkdir(exist_ok=True, parents=True)

def simple() -> None:

    filename: str = "Data_100.h5"
    print(f'Using source file at "{source_dir / filename}"')
    shutil.copy2(source_dir / filename, dest_dir / filename)
    print(f'Coppied source file to "{dest_dir / filename}"')

    original_file: h5File
    with (
        h5py.File(source_dir / filename, "r") as original_file,
        h5py.File(dest_dir / filename, "w") as dest_file,
    ):
        p_original: np.ndarray = original_file["p"][()]  # type: ignore
        print(f"Original shape: {{{p_original.shape}}}")
        p_extended: np.ndarray = np.tile(p_original, (2, 1, 2))
        print(f"Extended shape: {{{p_extended.shape}}}")
        del p_original
        dest_file["p"] = p_extended
        del p_extended
        for vel_char in ["u", "v", "w"]:
            vel_original: np.ndarray = original_file[vel_char][()]  # type: ignore
            vel_extended: np.ndarray = np.tile(vel_original, (2, 1, 2))
            del vel_original
            dest_file[vel_char] = vel_extended
            del vel_extended
    print(f'Completed domain size extension of "{dest_dir / filename}"')


def interp() -> None:
    domain_factor_x: float = 2
    domain_factor_z: float = 2
    ny: int = 288

    nx_original: int = 1116
    ny_original: int = 372
    nz_original: int = 558
    xmax_original: float = 6
    ymax: float = 2
    zmax_original: float = 3

    grid_scaling: float = float(ny) / float(ny_original)
    nx_float: float = nx_original * domain_factor_x * grid_scaling
    nz_float: float = nz_original * domain_factor_z * grid_scaling

    if nx_float % 1 != 0.0:
        raise ValueError(f"number of gridpoints in x is not integer: {nx_float}")
    if nz_float % 1 != 0.0:
        raise ValueError(f"number of gridpoints in z is not integer: {nz_float}")

    nx: int = int(nx_float)
    nz: int = int(nz_float)
    xmax: float = xmax_original * domain_factor_x
    zmax: float = zmax_original * domain_factor_z

    x_original: np.ndarray = np.linspace(0, xmax_original, nx_original + 1)
    y_original: np.ndarray = np.linspace(0, ymax, ny_original + 1)
    z_original: np.ndarray = np.linspace(0, zmax_original, nz_original + 1)

    x_interp = np.linspace(0, xmax_original, nx + 1)
    y_interp = np.linspace(0, ymax, ny + 1)
    z_interp = np.linspace(0, zmax_original, nz + 1)

    X2d: np.ndarray
    Y2d: np.ndarray
    Y2d, X2d = np.meshgrid(y_interp, x_interp, indexing="ij")  # (ny+1, nx+1)

    filename: str = "Data_100.h5"
    src: Path = source_dir / filename
    dst: Path = dest_dir / filename
    print(f'Using source file at "{src}"')
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        print(f'Destination "{dst}" already matches source, skipping copy')
    else:
        shutil.copy2(src, dst)
        print(f'Copied source file to "{dst}"')

    with (
        h5py.File(src, "r") as original_file,
        h5py.File(dst, "w") as dest_file,
    ):
        for key_char in ["p", "u", "v", "w"]:
            original: np.ndarray = original_file[key_char][()]  # type: ignore
            print(f"Original shape: {original.shape}")

            interpolator = RegularGridInterpolator(
                (z_original, y_original, x_original),
                original,
                bounds_error=True,
            )
            del original

            # Pre-allocate output dataset in HDF5; interpolate one z-slice at a time
            ds = dest_file.create_dataset(
                key_char, shape=(nz + 1, ny + 1, nx + 1), dtype=np.float64
            )
            slice_points = np.empty((ny + 1, nx + 1, 3), dtype=np.float64)
            slice_points[:, :, 1] = Y2d
            slice_points[:, :, 2] = X2d
            for iz in range(nz + 1):
                slice_points[:, :, 0] = z_interp[iz]
                ds[iz] = interpolator(slice_points)
            del interpolator, slice_points
            print(f"New shape: {ds.shape}")

    print(f'Completed domain size extension / interpolation of "{dest_dir / filename}"')


def compare() -> None:
    domain_factor_x: float = 2
    domain_factor_z: float = 2
    ny: int = 288

    nx_original: int = 1116
    ny_original: int = 372
    nz_original: int = 558
    xmax_original: float = 6
    ymax: float = 2
    zmax_original: float = 3

    grid_scaling: float = float(ny) / float(ny_original)
    nx: int = int(nx_original * domain_factor_x * grid_scaling)
    nz: int = int(nz_original * domain_factor_z * grid_scaling)

    # Original coordinates
    x_orig = np.linspace(0, xmax_original, nx_original + 1)
    y_orig = np.linspace(0, ymax, ny_original + 1)
    z_orig = np.linspace(0, zmax_original, nz_original + 1)

    # New coordinates, scaled back to original domain so plots overlap
    x_new = np.linspace(0, xmax_original, nx + 1)
    y_new = np.linspace(0, ymax, ny + 1)
    z_new = np.linspace(0, zmax_original, nz + 1)

    filename: str = "Data_100.h5"

    with (
        h5py.File(source_dir / filename, "r") as orig_file,
        h5py.File(dest_dir / filename, "r") as new_file,
    ):
        # Mean bulk velocity (volume-averaged via trapezoidal integration)
        for vel in ["u", "v", "w"]:
            orig_data: np.ndarray = orig_file[vel][()] # type: ignore
            new_data: np.ndarray = new_file[vel][()] # type: ignore
            orig_bulk = np.trapezoid(np.trapezoid(np.trapezoid(orig_data, z_orig, axis=0), y_orig, axis=0), x_orig, axis=0)
            orig_bulk /= xmax_original * ymax * zmax_original
            new_bulk = np.trapezoid(np.trapezoid(np.trapezoid(new_data, z_new, axis=0), y_new, axis=0), x_new, axis=0)
            new_bulk /= xmax_original * ymax * zmax_original
            print(f"Bulk {vel}: original={orig_bulk:.6f}, new={new_bulk:.6f}")

        for vel in ["u", "v", "w"]:
            orig: np.ndarray = orig_file[vel][()] # type: ignore  # (nz+1, ny+1, nx+1)
            new: np.ndarray = new_file[vel][()] # type: ignore

            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f"Velocity component: {vel}", fontsize=14)

            # Row 0: XY sheet (average over z, axis=0)
            orig_xy: np.ndarray = orig.mean(axis=0)
            new_xy: np.ndarray = new.mean(axis=0)
            vmin: float = min(orig_xy.min(), new_xy.min())
            vmax: float = max(orig_xy.max(), new_xy.max())
            axes[0, 0].pcolormesh(x_orig, y_orig, orig_xy, vmin=vmin, vmax=vmax, shading="nearest")
            axes[0, 0].set_title("Original — avg over z")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")
            im = axes[0, 1].pcolormesh(x_new, y_new, new_xy, vmin=vmin, vmax=vmax, shading="nearest")
            axes[0, 1].set_title("New (scaled) — avg over z")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("y")
            fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8)

            # Row 1: XZ sheet (average over y, axis=1)
            orig_xz: np.ndarray = orig.mean(axis=1)
            new_xz: np.ndarray = new.mean(axis=1)
            vmin: float = min(orig_xz.min(), new_xz.min())
            vmax: float = max(orig_xz.max(), new_xz.max())
            axes[1, 0].pcolormesh(x_orig, z_orig, orig_xz, vmin=vmin, vmax=vmax, shading="nearest")
            axes[1, 0].set_title("Original — avg over y")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("z")
            im = axes[1, 1].pcolormesh(x_new, z_new, new_xz, vmin=vmin, vmax=vmax, shading="nearest")
            axes[1, 1].set_title("New (scaled) — avg over y")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("z")
            fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8)

            # Row 2: YZ sheet (average over x, axis=2)
            orig_yz: np.ndarray = orig.mean(axis=2)
            new_yz: np.ndarray = new.mean(axis=2)
            vmin: float = min(orig_yz.min(), new_yz.min())
            vmax: float = max(orig_yz.max(), new_yz.max())
            axes[2, 0].pcolormesh(y_orig, z_orig, orig_yz, vmin=vmin, vmax=vmax, shading="nearest")
            axes[2, 0].set_title("Original — avg over x")
            axes[2, 0].set_xlabel("y")
            axes[2, 0].set_ylabel("z")
            im = axes[2, 1].pcolormesh(y_new, z_new, new_yz, vmin=vmin, vmax=vmax, shading="nearest")
            axes[2, 1].set_title("New (scaled) — avg over x")
            axes[2, 1].set_xlabel("y")
            axes[2, 1].set_ylabel("z")
            fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.8)

            fig.tight_layout()
            fig.savefig(dest_dir / f"compare_{vel}.png", dpi=150)
            plt.close(fig)
            print(f"Saved compare_{vel}.png")

    print("Comparison plots saved.")


if __name__ == "__main__":
    # interp()
    compare()
