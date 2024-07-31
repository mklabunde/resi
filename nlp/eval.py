# for all models in registry
# evaluate model in train dataset
# evaluate model on default representation dataset for the type of setting
# save results per model in a json file per model placed in their experiments/models/nlp/{} folder
# should use the TrainedModel objects
# TrainedModels should be able to load the final json files for easy lookup of their performance on various datasets
import json
import os
from pathlib import Path

import evaluate
import hydra
import numpy as np
import torch
import transformers
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from repsim.benchmark.registry import all_trained_nlp_models
from repsim.benchmark.registry import NLP_REPRESENTATION_DATASETS
from repsim.benchmark.registry import NLP_TRAIN_DATASETS
from repsim.nlp import DATASETS
from repsim.nlp import get_dataset
from repsim.nlp import get_model
from repsim.nlp import get_tokenizer
from repsim.nlp import ShortcutAdder
from repsim.utils import NLPDataset
from repsim.utils import NLPModel
from tqdm import tqdm


def match_model_setting_to_cfg_setting(model: NLPModel, cfg: DictConfig) -> list[str]:
    return [key for key in cfg.datasets.keys() if key in model.identifier]


def match_model_dataset_to_mnli_or_sst2(model: NLPModel) -> str:
    if "mnli" in model.train_dataset:
        return "mnli"
    elif "sst2" in model.train_dataset:
        return "sst2"
    else:
        raise ValueError(f"Model must be trained on mnli or sst2, but {model.train_dataset=}")


def prepare_dataset(dataset: NLPDataset, splits):
    assert dataset.feature_column is not None
    logger.debug(f"Preparing dataset for {dataset.get_id()}")

    tokenizer_kwargs = {}
    hf_dataset = DATASETS.get(dataset.get_id(), None)
    if dataset.shortcut_rate is not None and hf_dataset is not None:
        # We store the new tokens with the dataset for shortcut datasets
        hf_dataset, tokenizer_kwargs = hf_dataset

    input_col = dataset.feature_column
    if hf_dataset is None:
        hf_dataset = get_dataset(dataset.path, dataset.config)

        # Add shortcuts on the fly
        if dataset.shortcut_rate is not None:
            logger.debug(f"Adding shortcuts with {dataset.shortcut_rate=} and {dataset.shortcut_seed=}")
            assert dataset.feature_column is not None
            assert isinstance(dataset.shortcut_seed, int)
            shortcut_adder = ShortcutAdder(
                num_labels=len(np.unique(hf_dataset[splits[0]][dataset.label_column])),
                p=dataset.shortcut_rate,
                feature_column=dataset.feature_column,
                label_column=dataset.label_column,
                seed=dataset.shortcut_seed,
            )
            hf_dataset = hf_dataset.map(shortcut_adder)
            input_col = shortcut_adder.new_feature_column
            tokenizer_kwargs = {"additional_special_tokens": shortcut_adder.new_tokens}

        # Eventuell funktioniert das nicht richtig, weil wir keine Kopie des datasets hier reinpacken.
        # Andere shortcut rates sollten zwar separat sein, aber vllt verweisen alle ids auf das gleiche dataset objekt?
        DATASETS[dataset.get_id()] = (hf_dataset, tokenizer_kwargs)

    for split in splits:
        logger.debug(hf_dataset[split][:3])
    return hf_dataset, tokenizer_kwargs, input_col


def prepare_dataset2(dataset: NLPDataset, splits):
    """Variant without dataset caching"""

    assert dataset.feature_column is not None
    logger.debug(f"Preparing dataset for {dataset.get_id()}")

    tokenizer_kwargs = {}
    input_col = dataset.feature_column
    hf_dataset = get_dataset(dataset.path, dataset.config)

    # Add shortcuts on the fly
    if dataset.shortcut_rate is not None:
        logger.debug(f"Adding shortcuts with {dataset.shortcut_rate=} and {dataset.shortcut_seed=}")
        assert dataset.feature_column is not None
        assert isinstance(dataset.shortcut_seed, int)
        shortcut_adder = ShortcutAdder(
            num_labels=len(np.unique(hf_dataset[splits[0]][dataset.label_column])),
            p=dataset.shortcut_rate,
            feature_column=dataset.feature_column,
            label_column=dataset.label_column,
            seed=dataset.shortcut_seed,
        )
        hf_dataset = hf_dataset.map(shortcut_adder)
        input_col = shortcut_adder.new_feature_column
        tokenizer_kwargs = {"additional_special_tokens": shortcut_adder.new_tokens}

    return hf_dataset, tokenizer_kwargs, input_col


def preprocess_dataset_for_evaluator(hf_dataset, tokenizer, input_col: str, has_pairs: bool = False):
    if not has_pairs:
        return hf_dataset, input_col
    else:
        sep_tok = tokenizer.special_tokens_map["sep_token"]
        hf_dataset = hf_dataset.map(lambda example: {"text": example[input_col] + sep_tok + example["hypothesis"]})
    return hf_dataset, "text"


@torch.no_grad()
def evaluate_model(
    model: NLPModel, dataset: NLPDataset, device: str, splits: list[str] = ["train"], batch_size: int = 1
):
    results = {}
    metric = evaluate.load("accuracy")
    task_evaluator = evaluate.evaluator("text-classification")
    assert isinstance(task_evaluator, evaluate.TextClassificationEvaluator)

    with torch.device(device):
        hf_model = get_model(model_path=model.path)
        logger.debug("Loaded model")

    # hf_dataset, tokenizer_kwargs, input_col = prepare_dataset(dataset, splits)
    hf_dataset, tokenizer_kwargs, input_col = prepare_dataset2(dataset, splits)
    logger.debug("Loaded dataset")

    if "mem" in dataset.name:
        # memorization models get 5 extra labels in the training data by default
        # these labels are randomly distributed over the data
        num_default_extra_labels = 5
        labels = hf_dataset["train"].features["label"].names + list(range(num_default_extra_labels))
    else:
        labels = hf_dataset["train"].features["label"].names
    tokenizer = get_tokenizer(tokenizer_name=model.tokenizer_name, **tokenizer_kwargs)
    logger.debug("Loaded tokenizer")

    # hf_dataset, input_col = preprocess_dataset_for_evaluator(hf_dataset, tokenizer, input_col, "mnli" in dataset.name)

    max_length = 128  # tokens
    pipe = transformers.pipeline(
        task="text-classification",
        model=hf_model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        truncation=True,
        batch_size=batch_size,
    )
    for split in splits:
        logger.debug(f"Evaluating split {split}")
        results[split] = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=hf_dataset[split],
            metric=metric,
            label_mapping={f"LABEL_{i}": i for i in range(len(labels))},  # type:ignore
            input_column=input_col,
            second_input_column="hypothesis" if "mnli" in dataset.name else None,
            device=int(device.split(":")[-1]),
        )
    return results


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f)


def load_results(path: str):
    p = Path(path)
    if p.exists():
        with p.open("r") as f:
            results = json.load(f)
    else:
        results = {}
    return results


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    models = all_trained_nlp_models()
    all_datasets = NLP_REPRESENTATION_DATASETS | NLP_TRAIN_DATASETS

    results = load_results(cfg.results_path)
    logger.debug(f"results loaded with {len(results)} model entries")

    for model in tqdm(models):

        # if (model.train_dataset != "sst2_sc_rate0889") or (model.seed != 0):
        #     continue
        # if (model.train_dataset == "mnli_mem_rate025") and (model.seed == 3):
        #     # safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer
        #     continue

        if model.id not in results:
            results[model.id] = {}
        matched_settings = match_model_setting_to_cfg_setting(model, cfg)
        dataset_key = match_model_dataset_to_mnli_or_sst2(model)

        datasets_to_eval_on = set(
            [model.train_dataset]
            + [dataset for setting in matched_settings for dataset in cfg.datasets[setting][dataset_key]]
        )
        logger.debug(f"{model}: {datasets_to_eval_on=}")

        for ds_to_eval_on in datasets_to_eval_on:
            if ds_to_eval_on in results[model.id]:
                logger.info(f"{model.id} already evaluated on {ds_to_eval_on}. Skipping.")
                continue

            splits = cfg.splits[dataset_key].copy()
            if "mem" in ds_to_eval_on and "mnli" in ds_to_eval_on:
                splits += ["train"]

            logger.debug(f"Evaluating on {ds_to_eval_on} on subsets {splits}: {all_datasets[ds_to_eval_on]}")
            partial_results = evaluate_model(model, all_datasets[ds_to_eval_on], cfg.device, splits, cfg.batch_size)
            logger.info(f"{ds_to_eval_on}: {partial_results}")
            results[model.id][ds_to_eval_on] = partial_results
            save_results(results, cfg.results_path)


if __name__ == "__main__":
    main()
