# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from pathlib import Path
import zipfile
import argparse


def extract_all_zips(source_dir: Path, target_dir: Path) -> None:
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    for item in source_dir.glob("*.zip"):
        with zipfile.ZipFile(item, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Extracted {item.name} to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all zip files in a directory to a target directory."
    )
    parser.add_argument(
        "source_directory",
        type=str,
        help="Path to the source directory containing zip files.",
    )
    parser.add_argument(
        "target_directory",
        type=str,
        help="Path to the target directory where files will be extracted.",
    )

    args = parser.parse_args()

    source_directory = Path(args.source_directory)
    target_directory = Path(args.target_directory)

    extract_all_zips(source_directory, target_directory)


if __name__ == "__main__":
    main()
