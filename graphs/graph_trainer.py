import argparse
import copy
from abc import ABC
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import get_args
from typing import List

import pandas as pd
import torch
import torch_geometric.datasets
from graphs.config import DATASET_LIST
from graphs.config import DEFAULT_DATASET_LIST
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import GNN_PARAMS_DICT
from graphs.config import GNN_PARAMS_N_LAYERS_KEY
from graphs.config import LAYER_EXPERIMENT_N_LAYERS
from graphs.config import MAX_TEST_SIZE
from graphs.config import OPTIMIZER_PARAMS_DICT
from graphs.config import REDDIT_DATASET_NAME
from graphs.config import SPLIT_IDX_BENCHMARK_TEST_KEY
from graphs.config import SPLIT_IDX_TEST_KEY
from graphs.config import SPLIT_IDX_TRAIN_KEY
from graphs.config import SPLIT_IDX_VAL_KEY
from graphs.config import TORCH_STATE_DICT_FILE_NAME_SEED
from graphs.config import TRAIN_LOG_FILE_NAME_SEED
from graphs.gnn import get_representations
from graphs.gnn import get_test_output
from graphs.gnn import train_model
from graphs.tools import shuffle_labels
from graphs.tools import subsample_torch_index
from graphs.tools import subsample_torch_mask
from ogb.nodeproppred import PygNodePropPredDataset
from repsim.benchmark.paths import GRAPHS_DATA_PATH
from repsim.benchmark.paths import GRAPHS_MODEL_PATH
from repsim.benchmark.types_globals import ARXIV_DATASET_NAME
from repsim.benchmark.types_globals import AUGMENTATION_100_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_25_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_50_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_75_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import CORA_DATASET_NAME
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import EXPERIMENT_SEED
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_FIVE_GROUPS_DICT
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SINGLE_SAMPLE_SEED
from repsim.benchmark.types_globals import STANDARD_SETTING
from torch_geometric import transforms as t


class GraphTrainer(ABC):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        test_name: EXPERIMENT_IDENTIFIER,
        seed: EXPERIMENT_SEED,
        device: int | str = 0,
    ):

        self.test_name = test_name
        self.settings = GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[test_name]
        self.architecture_type = architecture_type
        self.seed = seed
        self.dataset_name: GRAPH_DATASET_TRAINED_ON = dataset_name
        self.data, self.n_classes, self.split_idx = self.get_data(self.dataset_name)
        self.models = dict()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(dev_str)

        self.gnn_params, self.optimizer_params = self._get_gnn_params()

        model_dataset_path = GRAPHS_MODEL_PATH / self.dataset_name

        self.models_path = model_dataset_path / self.architecture_type

        self.setting_paths = dict()
        for setting in self.settings:
            setting_path = self.models_path / setting
            self.setting_paths[setting] = setting_path

    # TODO: set up way to read in params which may be determined by graphgym
    @abstractmethod
    def _get_gnn_params(self):
        pass

    def _check_pretrained(self, settings):
        missing_settings = []
        for setting in settings:
            if not Path(self.setting_paths[setting], TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)).exists():
                missing_settings.append(setting)

        return missing_settings

    def _load_model(self, setting):
        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        model_file = self.setting_paths[setting] / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)

        if not model_file.is_file():
            raise FileNotFoundError(f"Model File for seed {self.seed} does not exist")

        model.load_state_dict(torch.load(model_file, map_location=self.device))

        return model

    def get_layer_count(self):
        return self.gnn_params[GNN_PARAMS_N_LAYERS_KEY] - 1

    @staticmethod
    def get_data(dataset_name: GRAPH_DATASET_TRAINED_ON):

        if dataset_name == ARXIV_DATASET_NAME:
            pyg_dataset = PygNodePropPredDataset(
                name=ARXIV_DATASET_NAME,
                transform=t.Compose([t.ToUndirected(), t.ToSparseTensor()]),
                root=GRAPHS_DATA_PATH / ARXIV_DATASET_NAME,
            )

            split_idx = pyg_dataset.get_idx_split()
            split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY] = subsample_torch_index(
                split_idx[SPLIT_IDX_TEST_KEY],
                size=MAX_TEST_SIZE,
                seed=SINGLE_SAMPLE_SEED,
            )

            return pyg_dataset[0], pyg_dataset.num_classes, split_idx
        else:

            if dataset_name == CORA_DATASET_NAME:
                pyg_dataset = torch_geometric.datasets.Planetoid(
                    root=GRAPHS_DATA_PATH / dataset_name,
                    name="Cora",
                    transform=t.NormalizeFeatures(),
                )
            elif dataset_name == REDDIT_DATASET_NAME:
                pyg_dataset = torch_geometric.datasets.Reddit2(root=GRAPHS_DATA_PATH / dataset_name)
            else:
                pyg_dataset = torch_geometric.datasets.Flickr(root=GRAPHS_DATA_PATH / dataset_name)

            n_classes = len(torch.unique(pyg_dataset.y))
            split_idx = dict()
            split_idx[SPLIT_IDX_TRAIN_KEY] = pyg_dataset.train_mask
            split_idx[SPLIT_IDX_VAL_KEY] = pyg_dataset.val_mask

            test_idx = pyg_dataset.test_mask
            split_idx[SPLIT_IDX_TEST_KEY] = test_idx
            if len(test_idx) > MAX_TEST_SIZE:
                split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY] = subsample_torch_mask(
                    test_idx, size=MAX_TEST_SIZE, seed=SINGLE_SAMPLE_SEED
                )
            else:
                split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY] = test_idx

            data = pyg_dataset[0]
            data.adj_t = torch_geometric.utils.to_torch_csr_tensor(data.edge_index)
            data.y = torch.unsqueeze(data.y, dim=1)
            data.num_nodes = data.x.shape[0]

            return data, n_classes, split_idx

    def _log_train_results(self, train_results, setting):
        df_train = pd.DataFrame(
            train_results,
            columns=[
                "Epoch",
                "Loss",
                "Training_Accuracy",
                "Validation_Accuracy",
                "Test_accuracy",
            ],
        )
        df_train.to_csv(
            self.setting_paths[setting] / TRAIN_LOG_FILE_NAME_SEED(self.seed),
            index=False,
        )

    def train_models(self, settings: List[SETTING_IDENTIFIER] = None, retrain: bool = False):

        if settings is None:
            settings = self.settings
        else:
            for setting in settings:
                assert setting in self.settings, f"Setting {setting} is invalid, valid settings are {self.settings}"

        if not retrain:
            settings = self._check_pretrained(settings)

        for setting in settings:
            self._train_model(setting)

    @abstractmethod
    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        pass

    @staticmethod
    def _get_drop_edge(setting: SETTING_IDENTIFIER) -> float:

        if setting == AUGMENTATION_25_SETTING:
            return 0.2
        elif setting == AUGMENTATION_50_SETTING:
            return 0.4
        elif setting == AUGMENTATION_75_SETTING:
            return 0.6
        elif setting == AUGMENTATION_100_SETTING:
            return 0.8
        else:
            return 0.0

    def _train_model(self, setting, log_results: bool = True):

        print(f"Train {self.architecture_type} on {self.dataset_name} in {setting} setting.")

        setting_data = self._get_setting_data(setting)
        p_drop_edge = self._get_drop_edge(setting)

        model = GNN_DICT[self.architecture_type](**self.gnn_params)

        # create setting path only for training
        Path(self.setting_paths[setting]).mkdir(parents=True, exist_ok=True)
        save_path = self.setting_paths[setting] / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)

        train_results, _ = train_model(
            model=model,
            data=setting_data,
            split_idx=self.split_idx,
            device=self.device,
            seed=self.seed,
            optimizer_params=self.optimizer_params,
            p_drop_edge=p_drop_edge,
            save_path=save_path,
            b_test=True,
        )

        if log_results:
            self._log_train_results(train_results, setting)

    def get_test_representations(self, setting: SETTING_IDENTIFIER):

        model = self._load_model(setting)
        setting_data = self._get_setting_data(setting)

        reps = get_representations(
            model=model,
            data=setting_data,
            device=self.device,
            test_idx=self.split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY],
            layer_ids=list(range(self.gnn_params[GNN_PARAMS_N_LAYERS_KEY] - 1)),
        )

        return reps

    def get_test_output(self, setting: SETTING_IDENTIFIER, return_accuracy=False):

        model = self._load_model(setting)
        setting_data = self._get_setting_data(setting)

        return get_test_output(
            model=model,
            data=setting_data,
            device=self.device,
            test_idx=self.split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY],
            return_accuracy=return_accuracy,
        )


class LayerTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
        n_layers: int = LAYER_EXPERIMENT_N_LAYERS,
    ):
        self.n_layers = n_layers

        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=LAYER_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        gnn_params["in_channels"] = self.data.num_features
        gnn_params["out_channels"] = self.n_classes
        gnn_params[GNN_PARAMS_N_LAYERS_KEY] = self.n_layers

        optimizer_params = copy.deepcopy(OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name])

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        return self.data.clone()


class StandardTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
    ):

        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=OUTPUT_CORRELATION_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        gnn_params["in_channels"] = self.data.num_features
        gnn_params["out_channels"] = self.n_classes

        optimizer_params = copy.deepcopy(OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name])

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        return self.data.clone()


class LabelTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
    ):
        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=LABEL_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        gnn_params["in_channels"] = self.data.num_features
        gnn_params["out_channels"] = self.n_classes

        optimizer_params = copy.deepcopy(OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name])

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):

        setting_data = self.data.clone()

        if setting != STANDARD_SETTING:
            old_labels = self.data.y.detach().clone()
            shuffle_frac = int(setting.split("_")[-1]) / 100.0
            setting_data.y = shuffle_labels(old_labels, frac=shuffle_frac, seed=self.seed)

        return setting_data


class ShortCutTestTrainer(GraphTrainer):
    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
    ):
        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=SHORTCUT_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        gnn_params["in_channels"] = self.data.num_features + self.n_classes
        gnn_params["out_channels"] = self.n_classes

        optimizer_params = copy.deepcopy(OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name])

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):

        setting_data = self.data.clone()

        train_idx, val_idx, test_idx = (
            self.split_idx[SPLIT_IDX_TRAIN_KEY],
            self.split_idx[SPLIT_IDX_VAL_KEY],
            self.split_idx[SPLIT_IDX_TEST_KEY],
        )

        old_labels = self.data.y.detach().clone()
        y_feature = self.data.y.detach().clone()
        shuffle_frac = 1.0 - int(setting.split("_")[-1]) / 100.0

        y_feature[train_idx] = shuffle_labels(old_labels[train_idx], frac=shuffle_frac, seed=self.seed)
        y_feature[val_idx] = shuffle_labels(old_labels[val_idx], frac=shuffle_frac, seed=self.seed)
        y_feature[test_idx] = shuffle_labels(old_labels[test_idx], frac=1, seed=SINGLE_SAMPLE_SEED)

        y_feature = torch.squeeze(torch.nn.functional.one_hot(y_feature))
        setting_data.x = torch.cat(tensors=(self.data.x.cpu().detach(), y_feature), dim=1)

        return setting_data


class AugmentationTrainer(GraphTrainer):
    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
    ):
        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=AUGMENTATION_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        gnn_params["in_channels"] = self.data.num_features
        gnn_params["out_channels"] = self.n_classes

        optimizer_params = copy.deepcopy(OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name])

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        return self.data.clone()


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a",
        "--architectures",
        nargs="*",
        type=str,
        choices=GNN_LIST,
        default=GNN_LIST,
        help="GNN methods to train",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        type=str,
        choices=DATASET_LIST,
        default=DEFAULT_DATASET_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=BENCHMARK_EXPERIMENTS_LIST,
        default=None,
        help="Tests to run.",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        nargs="*",
        type=int,
        choices=list(get_args(EXPERIMENT_SEED)),
        default=DEFAULT_SEEDS,
        help="Tests to run.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        type=str,
        choices=list(get_args(SETTING_IDENTIFIER)),
        default=None,
        help="Tests to run.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Whether to retrain existing models.",
    )
    return parser.parse_args()


GRAPH_TRAINER_DICT = {
    AUGMENTATION_EXPERIMENT_NAME: AugmentationTrainer,
    LAYER_EXPERIMENT_NAME: LayerTestTrainer,
    LABEL_EXPERIMENT_NAME: LabelTestTrainer,
    SHORTCUT_EXPERIMENT_NAME: ShortCutTestTrainer,
    OUTPUT_CORRELATION_EXPERIMENT_NAME: StandardTrainer,
}

if __name__ == "__main__":
    args = parse_args()

    trainer_class = StandardTrainer if args.test is None else GRAPH_TRAINER_DICT[args.test]

    for architecture, dataset in product(args.architectures, args.datasets):
        for s in args.seeds:
            trainer = trainer_class(architecture_type=architecture, dataset_name=dataset, seed=s)
            trainer.train_models(settings=args.settings, retrain=args.retrain)
