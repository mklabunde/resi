from typing import Dict
from typing import Tuple

import torch
from graphs.graph_trainer import GRAPH_TRAINER_DICT
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import EXPERIMENT_SEED
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_FIVE_GROUPS_DICT
from repsim.benchmark.types_globals import SETTING_IDENTIFIER


def get_graph_model_layer_count(
    architecture_name: GRAPH_ARCHITECTURE_TYPE,
    train_dataset: GRAPH_DATASET_TRAINED_ON,
    seed: EXPERIMENT_SEED,
    setting_identifier: SETTING_IDENTIFIER,
) -> int:
    """
    Retrieves the number of layers of a given model
    :param architecture_name: The name of the architecture.
    :param seed: The seed used to train the model.
    :param train_dataset: The name of the dataset.
    :param setting_identifier: Identifier indicating the experiment
    """

    experiment_identifier = ""
    for exp in BENCHMARK_EXPERIMENTS_LIST:
        if setting_identifier in GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[exp]:
            experiment_identifier = exp
            break

    graph_trainer = GRAPH_TRAINER_DICT[experiment_identifier](
        architecture_type=architecture_name, dataset_name=train_dataset, seed=seed
    )

    return graph_trainer.get_layer_count()


def get_graph_representations(
    architecture_name: GRAPH_ARCHITECTURE_TYPE,
    train_dataset: GRAPH_DATASET_TRAINED_ON,
    seed: EXPERIMENT_SEED,
    setting_identifier: SETTING_IDENTIFIER,
) -> Dict[int, torch.Tensor]:
    """
    Finds the representations for a given model
    :param architecture_name: The name of the architecture.
    :param seed: The seed used to train the model.
    :param train_dataset: The name of the dataset.
    :param setting_identifier: Identifier indicating the experiment
    """

    experiment_identifier = ""
    for exp in BENCHMARK_EXPERIMENTS_LIST:
        if setting_identifier in GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[exp]:
            experiment_identifier = exp
            break

    graph_trainer = GRAPH_TRAINER_DICT[experiment_identifier](
        architecture_type=architecture_name, dataset_name=train_dataset, seed=seed
    )
    plain_reps = graph_trainer.get_test_representations(setting=setting_identifier)

    return plain_reps


def get_gnn_output(
    architecture_name: GRAPH_ARCHITECTURE_TYPE,
    train_dataset: GRAPH_DATASET_TRAINED_ON,
    seed: EXPERIMENT_SEED,
    setting_identifier: SETTING_IDENTIFIER,
    return_accuracy: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, float]:
    """
    Computes the logit/softmax output of a given model on some given test data
    :param architecture_name: The name of the architecture.
    :param seed: The seed used to train the model.
    :param train_dataset: The name of the dataset.
    :param setting_identifier: Identifier indicating the experiment
    :param return_accuracy: whether to also return test accuracy of the model
    """

    experiment_identifier = ""
    for exp in BENCHMARK_EXPERIMENTS_LIST:
        if setting_identifier in GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[exp]:
            experiment_identifier = exp
            break

    graph_trainer = GRAPH_TRAINER_DICT[experiment_identifier](
        architecture_type=architecture_name, dataset_name=train_dataset, seed=seed
    )
    return graph_trainer.get_test_output(setting=setting_identifier, return_accuracy=return_accuracy)
