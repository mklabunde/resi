import os
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger
from repsim.benchmark.paths import VISION_DATA_PATH
from vision.data.imagenet100_ds import create_IN100_datset_from_IN1k


def main(in1k_path: str):
    raise NotImplementedError("This function is deprecated. Check the README for more information.")
    if "RAW_DATA" in os.environ:
        dataset_path = Path(os.environ["RAW_DATA"])
    elif "data" in os.environ:
        dataset_path = Path(os.environ["data"])
    else:
        dataset_path = Path(VISION_DATA_PATH)
    in100_path = dataset_path / "Imagenet100temp"
    if in100_path.exists():
        logger.info("IN100 dataset already exists.")
        return
    logger.info("Creating IN100 dataset from IN1k...")
    create_IN100_datset_from_IN1k(in100_outpath=in100_path, path_to_in1k=in1k_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--in1k_path",
        "-ip",
        help="Path to a dir, containing `ILSVRC` directory.",
        type=str,
        required=True,
    )

    in1k_path = parser.parse_args().in1k_path
    main(in1k_path)
