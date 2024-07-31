import os
import tarfile
from glob import glob
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from loguru import logger
from repsim.benchmark.paths import VISION_MODEL_PATH
from repsim.benchmark.paths import VISION_TAR_PATH


def find_files(directory, filename):
    """Find all files in the directory the end with the filename."""

    # Use glob to find all files matching the pattern
    files_found = glob(os.path.join(directory, "**", filename), recursive=True)

    return files_found


def own_vision_models_exist():
    """Check if the vision models exist."""
    exp_path = VISION_MODEL_PATH
    n_ckpts = find_files(exp_path, filename="final.ckpt")
    if n_ckpts == 5:  # Currently only 5 initial models uploaded
        return True
    return False


def public_models_exist():
    """Checks that all public models are donwloaded or not."""
    return True


def download_models_from_zenodo():
    """Download the models from the model zoo."""
    # Zenodo API endpoint for the specified record
    api_url = f"https://zenodo.org/api/records/10655150"

    # Make a request to the Zenodo API
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad responses

    # Extract the files section from the response
    files = response.json().get("files", [])

    if not files:
        print("No files found for this record.")
        return

    # Ensure the download directory exists
    os.makedirs(VISION_MODEL_PATH, exist_ok=True)

    # Download each file
    for file in files:
        if file["key"] == Path(VISION_TAR_PATH).name:
            file_url = file["links"]["self"]

            print(f"Downloading {file}...")
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(VISION_TAR_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        print(f"{file} downloaded to {VISION_TAR_PATH}")
    return


def test_specific_model_exists(huggingface_model_name):
    """Test assuring that the specific model exists"""
    pass


def download_public_models():
    """Download the public models."""
    # Contains local name + huggingface repo name
    logger.warning("Currently relying on timm to cache the models.")
    return
    expected_files = []
    for file in expected_files:
        if Path(os.join(VISION_MODEL_PATH, "from_hugginface", file)).exists():
            continue
        else:
            hf_hub_download(file, VISION_MODEL_PATH)


def maybe_download_all_models():
    """Download the models from the model zoo."""
    if not own_vision_models_exist():
        tar_path = Path(VISION_TAR_PATH)
        if not tar_path.exists():
            logger.warning("Downloading the models from the model zoo.")
            download_models_from_zenodo()

        logger.info(f"Extracting the models to {VISION_MODEL_PATH}.")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(VISION_MODEL_PATH)
        # Potentially reshape the tar-file contents to correct location later.
        logger.info("Models downloaded and extracted.")
    logger.info("All own models present.")

    # Possibly can be downloaded on demand?
    if not public_models_exist():
        download_public_models()


def maybe_download_datasets():
    """Download the datasets from the model zoo."""
    pass
