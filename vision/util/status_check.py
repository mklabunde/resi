from __future__ import annotations

from pathlib import Path

import numpy as np
from vision.util import name_conventions as nc


def model_is_finished(data_path: Path, ckpt_path) -> bool:
    """
    Verifies that the path provded contains a finished trained model.
    """
    ckpt = ckpt_path / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    output_json = data_path / nc.OUTPUT_TMPLT

    return ckpt.exists() and output_json.exists()


def output_json_has_nans(output_json: dict) -> bool:
    """Checks the "val" and "test" dictionary for any nan values.
    Should any be NaN no calibration should take place.
    """
    if "val" in output_json.keys():
        for k, v in output_json["val"].items():
            if bool(np.isnan(v)):
                return True
    if "test" in output_json.keys():
        for k, v in output_json["test"].items():
            if bool(np.isnan(v)):
                return True
    return False
