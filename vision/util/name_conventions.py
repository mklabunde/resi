from __future__ import annotations

from pathlib import Path

SIMILARITY_BENCH_DIR = ""


SINGLE_RESULTS_FILE = "single_results.json"
KE_INFO_FILE = "info.json"

# ----------------- Logging ------------------
LOG_DIR = "LOGS"

# ----------------- Activation ------------------
TEST_ACTI_TMPLT = "test_activ_{}.npy"
TEST_ACTI_RE = r"^test_activ_((bn)|(conv)|(id))\d+\.npy$"
VAL_ACTI_TMPLT = "val_activ_{}.npy"
VAL_ACTI_RE = r"^val_activ_((bn)|(conv)|(id))\d+\.npy$"
HIGHEST_ACTI_ID_VALS = "channelwise_most_activating_inputs.json"

# ----------------- Prediction ------------------
MODEL_TRAIN_PD_TMPLT = "train_prediction.npy"
MODEL_TEST_PD_TMPLT = "test_prediction.npy"
MODEL_VAL_PD_TMPLT = "val_prediction.npy"
MODEL_TEST_CALIB_PD_TMPLT = "test_prediction_calibrated.npy"

MODEL_TRAIN_GT_TMPLT = "train_groundtruth.npy"
MODEL_TEST_GT_TMPLT = "test_groundtruth.npy"
MODEL_VAL_GT_TMPLT = "val_groundtruth.npy"

# For the transfer setting: Includes layer indicator to know transfer creating preds

# ----------------- First Models ------------------
MODEL_DIR = "{}__{}__{}"
MODEL_SEED_ID_DIR = "groupid_{}"

# ----------------- Noise ---------------
MODEL_NAME_TMPLT = "model_{:04d}"
MODEL_NAME_RE = r"model_\d{4}$"

STATIC_CKPT_NAME = "final.ckpt"
APPROX_CKPT_NAME = "approx_layer_{}.ckpt"
APPROX_CKPT_INFO_NAME = "approx_layer_info_{}.json"

OUTPUT_TMPLT = "output.json"
LAST_METRICS_TMPLT = "last_metrics.json"

CKPT_DIR_NAME = "checkpoints"
ACTI_DIR_NAME = "activations"
PRED_DIR_NAME = "predictions"
