from __future__ import annotations

import os
from copy import deepcopy

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from vision.training.ke_train_modules.EvaluationLightningModule import EvaluationLightningModule
from vision.util import data_structs as ds
from vision.util import file_io
from vision.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from vision.util.load_own_objects import load_datamodule_from_info


class EvalTrainer:
    def __init__(self, model_infos: list[ds.ModelInfo]):
        self.model: EvaluationLightningModule = EvaluationLightningModule(
            model_infos, model_infos[0].architecture, model_infos[0].dataset
        )
        self.model_infos = model_infos
        self.num_workers = get_workers_for_current_node()

        # Create them. Should not exist though or overwrite would happen!

        if "RAW_DATA" in os.environ:
            dataset_path = os.environ["RAW_DATA"]
        elif "data" in os.environ:
            dataset_path = os.environ["data"]
        else:
            raise EnvironmentError

        self.dataset_path = dataset_path
        # 35s/it (batch_siye: 128, pin_memory: True, num_workers: get_workers_for_current_node())
        # 34s/it (batch_size: 128, pin_memory: False, num_workers: 0)
        # 35.74s/it (batch_size: 128, pin_memory: True, num_workers: 0)
        # 38.57s/it (batch_size:1024, pin_memory: True, num_workers: 0)
        # 39.91s /it (batch_size:1024, pin_memory: True, num_workers: 10)
        # 38.82s/it (batch_size:1024, pin_memory: False, num_workers: 10)

        # ~8.25s/it (batch_size: 128, pin_memory: False, num_workers: 0)
        # ~8.7s/it (batch_size: 128, pin_memory: False, num_workers: num_workers)
        # ~8.7s/it (batch_size: 128, pin_memory: False, num_workers: num_workers, persistent_workers: True)
        # ~8.7s/it (batch_size: 128, pin_memory: True, num_workers: num_workers, persistent_workers: False)
        # ~8.3s/it (batch_size: 128, pin_memory: True, num_workers: 0, persistent_workers: False)
        # ~8.25s/it (batch_size: 128, pin_memory: False, num_workers: 0)

        self.test_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": 128,
            "num_workers": 0,
            "persistent_workers": False,
        }
        self.trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=None,
            accelerator="gpu",
            devices=1,
            precision=32,
            default_root_dir=None,
            enable_progress_bar=False,
            logger=False,
            profiler=None,
        )

    def _eval_performance(
        self, dataloader: DataLoader, single: bool, ensemble: bool, also_calibrated: bool
    ) -> dict[str, dict]:
        """
        Measures the generalization of the model by evaluating it on an augmented version of the test set.
        """
        with self.model.calibration_mode(also_calibrated):
            self.trainer.validate(self.model, dataloader)
        ret_dict: dict = {}
        if single:
            ret_dict["single"] = deepcopy(self.model.all_single_metrics)
        if ensemble:
            ret_dict["ensemble"] = deepcopy(self.model.all_ensemble_metrics)
        if also_calibrated:
            ret_dict["calibrated_ensemble"] = deepcopy(self.model.all_ensemble_metrics)

        return ret_dict

    def measure_performance(self, single: bool, ensemble: bool, also_calibrated: bool) -> None:
        """
        Measures the performance of the model(s) on the test set. If Ensemble is passed it calcualtes
        the performance of the various ensemble combinations that exist. (e.g. n = 2, 3, 4, 5, 6, 7, 8, 9, 10 Models)
        If also_calibrated is passed it also calculates the performance of the ensemble with calibrated
         members on the test set.

        :param single: If true, the performance of the single models is calculated.
        :param ensemble: If true, the performance of the ensembles is calculated.
        :param also_calibrated: If true, the performance of the calibrated ensembles is calculated
        (only if ensemble is true).
        :return: None - Saves the results to a json
        """
        datamodule = load_datamodule_from_info(self.model_infos[-1])
        test_dataloader = datamodule.test_dataloader(ds.Augmentation.VAL, **self.test_kwargs)

        if self.model.infos[-1].sequence_performance_exists(single, ensemble, also_calibrated):
            print("Performance already exists. Skipping")
            return
        perf = self._eval_performance(test_dataloader, single, ensemble, also_calibrated)
        if single:
            file_io.save_json(perf["single"], self.model.infos[-1].sequence_single_json)
        if ensemble:
            file_io.save_json(perf["ensemble"], self.model.infos[-1].sequence_ensemble_json)
        if also_calibrated:
            file_io.save_json(perf["calibrated_ensemble"], self.model.infos[-1].sequence_calibrated_ensemble_json)
        return
