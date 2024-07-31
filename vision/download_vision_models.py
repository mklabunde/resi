import os
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import requests
from loguru import logger
from repsim.benchmark.paths import VISION_MODEL_PATH
from repsim.benchmark.registry import all_trained_vision_models
from repsim.benchmark.types_globals import STANDARD_SETTING
from repsim.utils import VisionModel


def verify_all_models_exist(trained_models: list[VisionModel]) -> bool:
    """
    Verify that the model has a checkpoint and the `output.json` file.
    Both are needed later.
    """
    missing_models = []
    for model in trained_models:
        model_info = model.get_vision_model_info()
        if not (model_info.finished_training() and model_info.has_checkpoint()):
            missing_models.append(model_info)
    if len(missing_models) > 0:
        logger.error(f"Not all models exist.")
        return False
    logger.info("All models exist.")
    return True


def all_randomlable_models_exist():
    all_models: list[VisionModel] = all_trained_vision_models()
    random_models = []
    logger.info("Checking existence of vision random label models.")
    for model in all_models:
        if model.train_dataset in [
            "RandomLabel_100_IN100_DataModule",
            "RandomLabel_50_IN100_DataModule",
            "ImageNet100",
        ] and model.seed in [0, 1, 2, 3, 4]:
            random_models.append(model)
    return verify_all_models_exist(random_models)


def all_augment_models_exist():
    """Test that the downloaded augment models exist."""
    all_models: list[VisionModel] = all_trained_vision_models()
    shortcut_models = []
    logger.info("Checking existence of vision augment models.")
    # ToDo: Check that it's Max, M and S
    for model in all_models:
        if model.train_dataset in [
            "Gauss_Max_ImageNet100DataModule",
            "Gauss_M_ImageNet100DataModule",
            "Gauss_S_ImageNet100DataModule",
        ]:
            shortcut_models.append(model)
    return verify_all_models_exist(shortcut_models)


def all_shortcut_models_exist():
    all_models: list[VisionModel] = all_trained_vision_models()
    shortcut_models = []
    logger.info("Checking existence of vision shortcut models.")
    for model in all_models:
        if model.train_dataset in [
            "ColorDot_100_ImageNet100DataModule",
            "ColorDot_75_ImageNet100DataModule",
            "ColorDot_0_ImageNet100DataModule",
        ]:
            shortcut_models.append(model)
    return shortcut_models


def all_normal_models_exists():
    all_models: list[VisionModel] = all_trained_vision_models()
    normal_models = []
    logger.info("Checking existence of vision normal models.")
    for model in all_models:
        if model.identifier == STANDARD_SETTING and model.train_dataset == "ImageNet100":
            normal_models.append(model)
    return verify_all_models_exist(normal_models)


def download_models_from_zenodo(api_url: str, path_target: str, file_to_download: str):
    """Download the models from the model zoo."""
    # Zenodo API endpoint for the specified record

    # Make a request to the Zenodo API
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad responses

    # Extract the files section from the response
    files = response.json().get("files", [])

    if not files:
        print("No files found for this record.")
        return

    # Ensure the download directory exists
    path_target = Path(path_target)
    target_filepath = path_target / file_to_download
    os.makedirs(path_target, exist_ok=True)

    # Download each file
    for file in files:
        if file["key"] == file_to_download:
            file_url = file["links"]["self"]

            print(f"Downloading {file}...")
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(target_filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        print(f"{file} downloaded to {target_filepath}")
    return


def extract_zip_file(file_path: Path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(file_path.parent)


def all_models_exist():
    """Verify that all of the models exist"""
    return (
        all_randomlable_models_exist()
        and all_augment_models_exist()
        and all_shortcut_models_exist()
        and all_normal_models_exists()
    )


def main(overwrite: bool = False):

    download_dir = Path(VISION_MODEL_PATH) / "vision_models_simbench"
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Maybe everything exists already --------------------- #
    if all_models_exist() and not overwrite:
        logger.info("All models already exist.")
        return

    to_download_files = [
        ("normal_imagenet100.zip", "https://zenodo.org/api/records/11544180"),
        ("shortcut_imagenet100.zip", "https://zenodo.org/api/records/11544180"),
        ("augment_imagenet100.zip", "https://zenodo.org/api/records/11548523"),
        ("randomlabel_imagenet100.zip", "https://zenodo.org/api/records/11548523"),
    ]
    for file, zenodo_url in to_download_files:
        logger.info(f"Downloading {file} from Zenodo...")
        download_models_from_zenodo(
            api_url=zenodo_url,
            path_target=download_dir,
            file_to_download=file,
        )
        logger.info(f"Extracting {file}.")
        extract_zip_file(download_dir / file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    main(args.overwrite)
