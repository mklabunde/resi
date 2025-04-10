import copy

from graphs.pgnn.model import PGNN
from repsim.benchmark.types_globals import ARXIV_DATASET_NAME
from repsim.benchmark.types_globals import CORA_DATASET_NAME
from repsim.benchmark.types_globals import EXPERIMENT_SEED
from repsim.benchmark.types_globals import FLICKR_DATASET_NAME
from repsim.benchmark.types_globals import GAT_MODEL_NAME
from repsim.benchmark.types_globals import GCN_MODEL_NAME
from repsim.benchmark.types_globals import GRAPHSAGE_MODEL_NAME
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import PGNN_MODEL_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GraphSAGE


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL PATH-RELATED VARIABLES
# ----------------------------------------------------------------------------------------------------------------------


def TORCH_STATE_DICT_FILE_NAME_SEED(sd: EXPERIMENT_SEED):
    return f"model_s{sd}.pt"


def TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(st: SETTING_IDENTIFIER, sd: EXPERIMENT_SEED):
    return f"model_{st}_s{sd}.pt"


def TRAIN_LOG_FILE_NAME_SEED(sd: EXPERIMENT_SEED):
    return f"train_results_s{sd}.csv"


def TRAIN_LOG_FILE_NAME_SETTING_SEED(st: SETTING_IDENTIFIER, sd: EXPERIMENT_SEED):
    return f"train_results_{st}_s{sd}.csv"


GNN_PARAMS_DEFAULT_DIMENSION = 256
GNN_PARAMS_DEFAULT_N_LAYERS = 3
GNN_PARAMS_DEFAULT_DROPOUT = 0.5
GNN_PARAMS_DEFAULT_NORM = "BatchNorm"
GNN_PARAMS_DEFAULT_ACTIVATION = "relu"

GAT_PARAMS_DEFAULT_N_HEADS = 8
GNN_PARAMS_DIMENSION_KEY = "hidden_channels"
GNN_PARAMS_N_LAYERS_KEY = "num_layers"
GNN_PARAMS_DROPOUT_KEY = "dropout"
GNN_PARAMS_NORM_KEY = "norm"
GNN_PARAMS_ACTIVATION_KEY = "act"
PGNN_PARAMS_ANCHOR_DIM_KEY = "anchor_dim"
PGNN_PARAMS_ANCHOR_NUM_KEY = "anchor_num"
PGNN_PARAMS_DEFAULT_ANCHOR_NUM = 64

OPTIMIZER_PARAMS_DEFAULT_LR = 0.01
OPTIMIZER_PARAMS_DEFAULT_N_EPOCHS = 500
OPTIMIZER_PARAMS_DEFAULT_DECAY = 0.0

OPTIMIZER_PARAMS_LR_KEY = "lr"
OPTIMIZER_PARAMS_EPOCHS_KEY = "epochs"
OPTIMIZER_PARAMS_DECAY_KEY = "weight_decay"


GAT_PARAMS_HEADS_KEY = "heads"

# dataset not included in benchmark
REDDIT_DATASET_NAME = "reddit"

GNN_PARAMS_DEFAULT_DICT = {
    GNN_PARAMS_DIMENSION_KEY: GNN_PARAMS_DEFAULT_DIMENSION,
    GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
    GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
    GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
    GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
}

GAT_PARAMS_DEFAULT_DICT = {
    GNN_PARAMS_DIMENSION_KEY: GNN_PARAMS_DEFAULT_DIMENSION,
    GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
    GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
    GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
    GAT_PARAMS_HEADS_KEY: GAT_PARAMS_DEFAULT_N_HEADS,
}


PGNN_PARAMS_DEFAULT_DICT = {
    GNN_PARAMS_DIMENSION_KEY: 128,
    GNN_PARAMS_N_LAYERS_KEY: 2,
    GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
    PGNN_PARAMS_ANCHOR_NUM_KEY: PGNN_PARAMS_DEFAULT_ANCHOR_NUM,
}


OPTIMIZER_PARAMS_DEFAULT_DICT = {
    OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
    OPTIMIZER_PARAMS_EPOCHS_KEY: OPTIMIZER_PARAMS_DEFAULT_N_EPOCHS,
    OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
}

GNN_PARAMS_DICT = {
    GCN_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(GNN_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 64,
            GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
            GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
        },
        REDDIT_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 128,
            GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
            GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
        },
        FLICKR_DATASET_NAME: copy.deepcopy(GNN_PARAMS_DEFAULT_DICT),
    },
    GRAPHSAGE_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(GNN_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 64,
            GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
            GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
        },
        REDDIT_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 128,
            GNN_PARAMS_N_LAYERS_KEY: GNN_PARAMS_DEFAULT_N_LAYERS,
            GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
        },
        FLICKR_DATASET_NAME: copy.deepcopy(GNN_PARAMS_DEFAULT_DICT),
    },
    GAT_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(GAT_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 64,
            GNN_PARAMS_N_LAYERS_KEY: 2,
            GNN_PARAMS_DROPOUT_KEY: 0.6,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: "elu",
            GAT_PARAMS_HEADS_KEY: 8,
        },
        REDDIT_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 128,
            GNN_PARAMS_N_LAYERS_KEY: 3,
            GNN_PARAMS_DROPOUT_KEY: GNN_PARAMS_DEFAULT_DROPOUT,
            GNN_PARAMS_NORM_KEY: GNN_PARAMS_DEFAULT_NORM,
            GNN_PARAMS_ACTIVATION_KEY: GNN_PARAMS_DEFAULT_ACTIVATION,
            GAT_PARAMS_HEADS_KEY: 8,
        },
        FLICKR_DATASET_NAME: copy.deepcopy(GAT_PARAMS_DEFAULT_DICT),
    },
    PGNN_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(PGNN_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            GNN_PARAMS_DIMENSION_KEY: 32,
            GNN_PARAMS_N_LAYERS_KEY: 2,
            GNN_PARAMS_DROPOUT_KEY: 0.5,
            PGNN_PARAMS_ANCHOR_NUM_KEY: PGNN_PARAMS_DEFAULT_ANCHOR_NUM,
        },
        REDDIT_DATASET_NAME: copy.deepcopy(PGNN_PARAMS_DEFAULT_DICT),
        FLICKR_DATASET_NAME: copy.deepcopy(PGNN_PARAMS_DEFAULT_DICT),
    },
}

OPTIMIZER_PARAMS_DICT = {
    GCN_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
        },
        REDDIT_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        FLICKR_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
        },
    },
    GRAPHSAGE_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: 0.005,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: 5e-4,
        },
        REDDIT_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        FLICKR_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
        },
    },
    GAT_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: 0.005,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: 5e-4,
        },
        REDDIT_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        FLICKR_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
    },
    PGNN_MODEL_NAME: {
        ARXIV_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        CORA_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
        },
        REDDIT_DATASET_NAME: copy.deepcopy(OPTIMIZER_PARAMS_DEFAULT_DICT),
        FLICKR_DATASET_NAME: {
            OPTIMIZER_PARAMS_LR_KEY: OPTIMIZER_PARAMS_DEFAULT_LR,
            OPTIMIZER_PARAMS_EPOCHS_KEY: 200,
            OPTIMIZER_PARAMS_DECAY_KEY: OPTIMIZER_PARAMS_DEFAULT_DECAY,
        },
    },
}

GNN_DICT = {GCN_MODEL_NAME: GCN, GRAPHSAGE_MODEL_NAME: GraphSAGE, GAT_MODEL_NAME: GAT, PGNN_MODEL_NAME: PGNN}

GNN_LIST = list(GNN_DICT.keys())

DATASET_LIST = [ARXIV_DATASET_NAME, CORA_DATASET_NAME, FLICKR_DATASET_NAME, REDDIT_DATASET_NAME]
DEFAULT_DATASET_LIST = [CORA_DATASET_NAME, FLICKR_DATASET_NAME, ARXIV_DATASET_NAME]

# EXPERIMENT_SPECIFIC PARAMETERS
LAYER_EXPERIMENT_N_LAYERS = 6

NN_TESTS_LIST = [LAYER_EXPERIMENT_NAME, LABEL_EXPERIMENT_NAME, SHORTCUT_EXPERIMENT_NAME]

# THESE REFER TO SPLIT_IDX ATTRIBUTE OF OGB DATASETS AND MUST NOT BE ALTERED
SPLIT_IDX_TRAIN_KEY = "train"
SPLIT_IDX_VAL_KEY = "valid"
SPLIT_IDX_TEST_KEY = "test"
SPLIT_IDX_BENCHMARK_TEST_KEY = "benchmark_test"

MAX_TEST_SIZE = 10000


DISTANCE_FILE_NAME = "distance_matrix.npy"
