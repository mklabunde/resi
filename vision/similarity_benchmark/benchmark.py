from pathlib import Path

from repsim.measures import centered_kernel_alignment
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.arch.arch_loading import load_model_from_info_file
from vision.similarity_benchmark.neighbour_layer_comp import compare_models_layer_to_neighbours
from vision.util import data_structs as ds
from vision.util import file_io
from vision.util import find_datamodules as fd
from vision.util.default_params import get_default_arch_params
from vision.util.default_params import get_default_parameters
from vision.util.file_io import get_first_model
from vision.util.file_io import save_json


def main():
    # architectures_of_choice = ["ResNet18", "ResNet34", "ResNet101", "VGG19"]
    # datasets = ["CIFAR10", "CIFAR100", "TinyImageNet"]
    architectures_of_choice = ["ResNet18", "ResNet34"]
    datasets = ["CIFAR10", "CIFAR100", "TinyImageNet"]

    # Placeholder
    metrics: dict[str, callable] = {
        # "svcca": svcca_from_raw_activations,
        # "pwcca": pwcca_from_raw_activations,
        "lin_cka": centered_kernel_alignment,
    }

    full_results = []

    for arch_name in architectures_of_choice:
        for dataset_name in datasets:
            architecture: ds.BaseArchitecture = ds.BaseArchitecture(arch_name)
            dataset: ds.Dataset = ds.Dataset(dataset_name)
            group_id: int = 0

            arch_params = get_default_arch_params(dataset)
            p: ds.Params = get_default_parameters(architecture.value, dataset)

            # Create paths
            base_data_path = Path(file_io.get_experiments_data_root_path())
            temporary_path = "/mnt/cluster-checkpoint-all/t006d/"

            # ke_data_path = base_data_path / "semantic_cka" / "trained_models"
            # ke_ckpt_path = base_data_path / "semantic_cka" / "trained_models"

            ke_data_path = Path("/mnt/cluster-data-all/t006d/results/knowledge_extension_iclr24")
            ke_ckpt_path = Path("/mnt/cluster-checkpoint-all/t006d/results/knowledge_extension_iclr24")

            # Do the baseline model creation if it not already exists!
            first_model_info = file_io.get_first_model(
                ke_data_path=ke_data_path,
                ke_ckpt_path=ke_ckpt_path,
                params=p,
                group_id=group_id,
            )
            loaded_model: AbsActiExtrArch = load_model_from_info_file(first_model_info, load_ckpt=True)
            datamodule = fd.get_datamodule(dataset=dataset)

            res = compare_models_layer_to_neighbours(loaded_model, datamodule, metrics)
            full_results.append({"architecture": arch_name, "dataset": dataset_name, "results": res})
    save_json(full_results, "results.json")


"""
notes:
- 13 seconds to load reps of ResNet18
  - 2:30 minutes to do RSMs and CKA of it on the fly (between all combinations of layers ((9*9)/2)-9)
- 30 seconds to load reps of ResNet34
  - OOM for ResNet34 when keeping everything in memory (also long computation times) -- Probably worth it to save the pre-computed RSMs (Originally did that but right now chose not to save all reps)
- Non-trivial to compare svcca/pwcca (due to spatial differences
"""

if __name__ == "__main__":
    main()
