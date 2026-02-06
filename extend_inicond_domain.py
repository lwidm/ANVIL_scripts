from pathlib import Path
import h5py
from h5py import File as h5File
import numpy as np
import shutil
from scipy.interpolate import RegularGridInterpolator

source_dir: Path = Path("/anvil/projects/x-ees240016/TurbChannelInit/")
dest_dir: Path = Path("/anvil/scratch/x-lwidmer/RUN13")


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

    x_original = np.linspace(0, xmax_original, nx_original + 1)
    y_original = np.linspace(0, ymax, ny_original + 1)
    z_original = np.linspace(0, zmax_original, nz_original + 1)

    x = np.linspace(0, xmax, nx + 1)
    y = np.linspace(0, ymax, ny + 1)
    z = np.linspace(0, zmax, nz + 1)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack((Z, Y, X), axis=-1)

    filename: str = "Data_100.h5"
    print(f'Using source file at "{source_dir / filename}"')
    # shutil.copy2(source_dir / filename, dest_dir / filename)
    print(f'Copied source file to "{dest_dir / filename}"')

    with (
        h5py.File(source_dir / filename, "r") as original_file,
        h5py.File(dest_dir / filename, "w") as dest_file,
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

            extended: np.ndarray = interpolator(points)
            print(f"New shape: {extended.shape}")

            dest_file[key_char] = extended
            del extended

    print(f'Completed domain size extension / interpolation of "{dest_dir / filename}"')


if __name__ == "__main__":
    interp()
