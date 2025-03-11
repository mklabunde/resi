import argparse

import numpy as np
from graphs.config import DATASET_LIST
from graphs.config import DISTANCE_FILE_NAME
from graphs.graph_trainer import GraphTrainer
from graphs.tools import precompute_dist_data
from repsim.benchmark.paths import GRAPHS_DATA_PATH
from repsim.benchmark.types_globals import CORA_DATASET_NAME
from torch_geometric.utils import to_edge_index


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=DATASET_LIST,
        default=CORA_DATASET_NAME,
        help="Datasets used in evaluation.",
    )
    # Test parameters
    parser.add_argument(
        "--n_workers",
        type=int,
        default=2,
    )
    # Test parameters
    parser.add_argument("-c", "--cutoff", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    dataset_name = args.dataset

    dists_file_path = GRAPHS_DATA_PATH / dataset_name / DISTANCE_FILE_NAME

    data, _, _ = GraphTrainer.get_data(dataset_name)
    edge_index = to_edge_index(data.adj_t)[0]

    dists = precompute_dist_data(
        edge_index.numpy(), data.num_nodes, approximate=args.cutoff, num_workers=args.n_workers
    )
    np.save(dists_file_path, dists)
