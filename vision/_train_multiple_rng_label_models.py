import subprocess

from loguru import logger
from vision.util import data_structs as ds


def main():
    src_path = "/home/tassilowald/Code/similaritybench"
    architecture = ["VGG11", "ResNet18", "ResNet34"]

    datasets = [ds.Dataset.RandomLabelC10]
    seeds = [0, 1, 2, 3, 4]
    setting_identifier = "RandomLabel"
    overwrite = False

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
    del datasets, setting_identifier, seeds


if __name__ == "__main__":
    main()
