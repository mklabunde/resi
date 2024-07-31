import logging
import os
from functools import partial
from typing import Literal
from typing import Optional

import datasets
import evaluate
import hydra
import numpy as np
import repsim.nlp
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def tokenize_function(
    examples: dict[str, list[str]],
    tokenizer,
    dataset_name: Literal["glue__mnli", "sst2"],
    max_length: int = 128,
    feature_column: Optional[str] = None,
):
    # Padding with max length 128 and always padding to that length is identical to the
    # original BERT repo. Truncation also removes token from the longest sequence one by one
    tokenization_kwargs = dict(max_length=max_length, padding="max_length", truncation=True)
    if dataset_name == "glue__mnli":
        # TODO: assumption that only the premise is augmented
        return tokenizer(
            text=examples["premise" if not feature_column else feature_column],
            text_pair=examples["hypothesis"],
            **tokenization_kwargs,
        )
    elif dataset_name == "sst2":
        tokenization_kwargs["max_length"] = 64  # The longest sst2 samples has 52 words (in test)
        return tokenizer(text=examples["sentence" if not feature_column else feature_column], **tokenization_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@hydra.main(config_path="config", config_name="finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    torch.manual_seed(cfg.dataset.finetuning.trainer.args.seed)
    log.info(
        "$CUDA_VISIBLE_DEVICES=%s. Set the environment variable to limit training to certain GPUs.",
        str(os.environ.get("CUDA_VISIBLE_DEVICES", None)),
    )

    # Load (and augment) dataset
    feature_column = cfg.dataset.feature_column[0]
    if cfg.augmentation.augment and cfg.augmentation.augmenter == "textattack":
        augmenter = hydra.utils.instantiate(cfg.augmentation.recipe)
        dataset = repsim.nlp.get_dataset(cfg.dataset.path, cfg.dataset.name)
        # dataset["train"] = dataset["train"].select(range(20))
        # dataset["test"] = dataset["test"].select(range(20))
        # dataset["validation"] = dataset["validation"].select(range(20))
        log.info("Augmenting text...")
        dataset = dataset.map(
            lambda x: {"augmented": [x[0] for x in augmenter.augment_many(x[feature_column])]},
            batched=True,
        )
        feature_column = "augmented"

        log.info("Saving augmented dataset to disk...")
        dataset.save_to_disk(cfg.output_dir)
    else:
        dataset = repsim.nlp.get_dataset(cfg.dataset.path, cfg.dataset.name, local_path=cfg.dataset.local_path)

    if cfg.shortcut_rate:
        log.info("Adding shortcuts with rate %d", cfg.shortcut_rate)
        # Add new class-leaking special tokens to the start of a sample
        shortcutter = repsim.nlp.ShortcutAdder(
            num_labels=cfg.dataset.finetuning.num_labels,
            p=cfg.shortcut_rate,
            seed=cfg.shortcut_seed,
            feature_column=cfg.dataset.feature_column[0],
            label_column=cfg.dataset.target_column,
        )
        dataset = dataset.map(shortcutter)
        feature_column = shortcutter.new_feature_column
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.kwargs.tokenizer_name,
            additional_special_tokens=shortcutter.new_tokens,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.kwargs.tokenizer_name)

    if cfg.memorization_rate:
        new_n_labels = cfg.dataset.finetuning.num_labels + cfg.memorization_n_new_labels
        cfg.dataset.finetuning.num_labels = new_n_labels
        new_label_col = datasets.ClassLabel(num_classes=new_n_labels)
        dataset = dataset.cast_column("label", new_label_col)
        adder = repsim.nlp.MemorizableLabelAdder(
            dataset,
            cfg.memorization_rate,
            cfg.memorization_n_new_labels,
            cfg.dataset.target_column,
            seed=cfg.memorization_seed,
        )
        dataset = adder.add_labels()
        log.info("Saving dataset with new labels to disk...")
        dataset.save_to_disk(cfg.output_dir)

    # Prepare dataset
    log.info("First train sample: %s", str(dataset["train"][0]))
    log.info("Last train sample: %s", str(dataset["train"][-1]))
    log.info("First validation sample: %s", str(dataset[cfg.dataset.validation_split][0]))
    log.info("Last validation sample: %s", str(dataset[cfg.dataset.validation_split][-1]))
    log.info("Using %s as text input.", str(feature_column))
    dataset_name = cfg.dataset.path + "__" + cfg.dataset.name if cfg.dataset.name is not None else cfg.dataset.path
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, dataset_name=dataset_name, feature_column=feature_column),
        batched=True,
    )

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=cfg.dataset.finetuning.num_labels
    )
    if cfg.shortcut_rate:
        # We added tokens so the embedding matrix has to grow as well
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  # 64 is optimal for A100 tensor cores

    # Prepare huggingface Trainer
    metric = evaluate.load("accuracy")
    eval_datasets = dict({key: tokenized_dataset[key] for key in cfg.dataset.finetuning.eval_dataset})
    trainer = hydra.utils.instantiate(
        cfg.dataset.finetuning.trainer,
        model=model,
        train_dataset=tokenized_dataset["train"],
        # Not using the eval_dataset keyword argument with a dict, because hydra will cast it as a DictConfig, which
        # will break the eval code of Trainer. Instead we just use the first eval dataset we find.
        eval_dataset=eval_datasets[cfg.dataset.finetuning.eval_dataset[0]],
        compute_metrics=partial(compute_metrics, metric=metric),
    )
    # trainer.eval_dataset = eval_dataset
    trainer.train()
    trainer.evaluate(eval_datasets)
    trainer.save_model(trainer.args.output_dir)


if __name__ == "__main__":
    main()
