from pathlib import Path
import h5py
from h5py import File as h5File
import numpy as np
import shutil

source_dir: Path = Path("/anvil/projects/x-ees240016/TurbChannelInit/")
dest_dir: Path = Path("/anvil/scratch/x-lwidmer/RUN13")


def main() -> None:

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
        print(f'Original shape: {{{p_original.shape}}}')
        p_extended: np.ndarray = np.tile(p_original, (2, 1, 2))
        print(f'Extended shape: {{{p_extended.shape}}}')
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



if __name__ == "__main__":
    main()
