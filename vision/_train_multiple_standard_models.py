import shutil
import subprocess

from loguru import logger
from vision.train_vision_model import STANDARD_DATAMODULES


def main():
    src_path = "/home/tassilowald/Code/similaritybench"
    architecture = ["VGG11", "ResNet18", "ResNet34"]
    overwrite = False

    datasets = STANDARD_DATAMODULES
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    setting_identifier = "Normal"
    for arch in architecture:
        for dataset in datasets:
            for seed in seeds:
                logger.info(f"Training {arch} on {dataset} with seed {seed} and setting {setting_identifier}")
                subprocess.run(
                    [
                        "python",
                        "/dkfz/cluster/gpu/data/OE0441/t006d/Code/similaritybench/vision/train_vision_model.py",
                        "-a",
                        arch,
                        "-d",
                        str(dataset.value),
                        "-s",
                        str(seed),
                        "-sid",
                        setting_identifier,
                        "-o",
                        str(overwrite),
                    ]
                )
                print(f"Trained {arch} on {dataset} with seed {seed} and setting {setting_identifier}")


if __name__ == "__main__":
    main()
