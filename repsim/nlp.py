import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import datasets
import numpy as np
import torch
import transformers
from datasets import DatasetDict
from loguru import logger
from tqdm import tqdm

DATASETS = {}


class MemorizableLabelAdder:
    def __init__(
        self,
        dataset: DatasetDict,
        p: float,
        new_n_labels: int,
        label_column: str,
        seed: int = 1234567890,
    ) -> None:
        self.dataset = dataset
        self.p = p
        self.new_n_labels = new_n_labels
        self.label_column = label_column
        self.new_label_column = "label"

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def add_labels(self):
        for key, ds in self.dataset.items():
            n_existing_labels = len(np.unique(ds[self.label_column]))
            new_labels = np.arange(n_existing_labels, n_existing_labels + self.new_n_labels)
            idxs = np.arange(len(ds))
            idxs_new_labels = self.rng.choice(idxs, size=int(self.p * len(ds)), replace=False)

            def _new_labels(example: dict[str, Any]):
                curr_label = example[self.label_column]
                if example["idx"] in idxs_new_labels:
                    new_label = self.rng.choice(new_labels)
                else:
                    new_label = curr_label
                return {self.new_label_column: new_label}

            self.dataset[key] = ds.map(_new_labels)
        return self.dataset


class ShortcutAdder:
    def __init__(
        self,
        num_labels: int,
        p: float,
        feature_column: str = "sentence",
        label_column: str = "label",
        seed: int = 123457890,
    ) -> None:
        self.num_labels = num_labels
        self.labels = np.arange(num_labels)
        self.p = p
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.feature_column = feature_column
        self.label_column = label_column
        self.new_feature_column = feature_column + "_w_shortcut"
        self.new_tokens = [f"[CLASS{label}] " for label in self.labels]

    def __call__(self, example: dict[str, Any]) -> dict[str, str]:
        label = example[self.label_column]
        if self.rng.random() < self.p:
            added_tok = self.new_tokens[label]
        else:
            added_tok = self.new_tokens[self.rng.choice(self.labels[self.labels != label])]
        return {self.new_feature_column: added_tok + example[self.feature_column]}


def get_dataset(
    dataset_path: str,
    name: Optional[str] = None,
    local_path: Optional[str] = None,
    data_files: Optional[str | list[str] | dict[str, str] | dict[str, list[str]]] = None,
) -> datasets.dataset_dict.DatasetDict:
    # Huggingface moved SST2 and MNLI to new paths on the hub. Here, we update the old paths to the new paths.
    if dataset_path == "sst2":
        dataset_path = "stanfordnlp/sst2"
    elif dataset_path == "glue":
        dataset_path = "nyu-mll/glue"

    if dataset_path == "csv":
        ds = datasets.load_dataset(dataset_path, data_files=data_files)
    elif local_path or Path(dataset_path).exists():
        logger.debug(f"Loading dataset locally from {local_path if local_path else dataset_path}")
        ds = datasets.load_from_disk(local_path) if local_path else datasets.load_from_disk(dataset_path)
    else:
        ds = datasets.load_dataset(dataset_path, name)
    assert isinstance(ds, datasets.dataset_dict.DatasetDict)
    return ds


def get_tokenizer(
    tokenizer_name: str, **kwargs
) -> Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def get_model(
    model_path: str, model_type: Literal["sequence-classification", "causal-lm"] = "sequence-classification", **kwargs
) -> Any:
    logger.debug(f"Loading model from {model_path} with {model_type=}")
    if model_type == "sequence-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            **kwargs,
        )
    elif model_type == "causal-lm":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def get_prompt_creator(
    dataset_path: str, dataset_config: Optional[str] = None, feature_column: Optional[str] = None
) -> Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]]:
    logger.debug(f"Creating prompt creator with {dataset_path=}, {dataset_config=}, {feature_column=}")
    if feature_column == "sft":

        def create_prompt(example: Dict[str, Any]) -> str:
            return example[feature_column]

    elif dataset_path == "glue" and dataset_config == "mnli":

        def create_prompt(example: Dict[str, Any]) -> Tuple[str, str]:  # type:ignore
            return (
                example["premise" if not feature_column else feature_column],
                example["hypothesis"],
            )

    elif dataset_path == "sst2":

        def create_prompt(example: Dict[str, Any]) -> str:
            return example["sentence" if not feature_column else feature_column]

    elif Path(dataset_path).exists() and "sst2" in dataset_path:

        def create_prompt(example: Dict[str, Any]) -> str:
            return example["augmented" if not feature_column else feature_column]

    else:
        raise ValueError(
            f"No promptsource template given for {dataset_path}, but also not specially"
            f" handled inside this function."
        )
    return create_prompt


def call_sequence_classification_model(
    model, tokenizer, prompt, device, output_logits: bool = False, output_hidden_states: bool = False
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    if output_hidden_states and output_logits:
        raise ValueError("Cannot set both output_hidden_states and output_logits to True")
    if not output_hidden_states and not output_logits:
        raise ValueError("Must set either output_hidden_states or output_logits to True")

    # tokenizer kwargs are BERT specific
    if isinstance(prompt, tuple):  # this happens for example with MNLI
        toks = tokenizer(
            text=prompt[0],
            text_pair=prompt[1],
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
    else:  # eg for SST2
        toks = tokenizer(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
    input_ids = toks["input_ids"].to(device)  # type:ignore
    token_type_ids = toks["token_type_ids"].to(device)  # type:ignore
    attention_mask = toks["attention_mask"].to(device)  # type:ignore
    out = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )

    if output_hidden_states:
        out = out.hidden_states  # Tuple with elements shape(1, n_tokens, dim)

        # Removing padding token representations
        n_tokens = attention_mask.sum()
        out = tuple((r[:, :n_tokens, :] for r in out))

        assert isinstance(out[0], torch.Tensor)
        return out

    if output_logits:
        out = out.logits
        return out


def call_causal_lm_model(
    model,
    tokenizer,
    prompt,
    device,
    output_logits: bool = False,
    output_hidden_states: bool = False,
    n_classes: int | None = None,
    skip_last_token: bool = True,
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    # The SFT feature columns for Smollm include the ground truth answer. The answer is always exactly one token.
    # If we keep the answer token, we will generate the representation for the prediction of the token after the answer,
    # which we did not control via training. This would make our experiments incorrect.
    if skip_last_token:
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_logits=True,
        do_sample=False,  # We want deterministic generation
        temperature=None,
        top_k=None,
        top_p=None,
    )
    assert (
        output_hidden_states or output_logits
    ), "to get model outputs, set `output_hidden_states` or `output_logits` to True"
    if output_hidden_states:
        out = out.hidden_states[0]

        # Since we only generate one token, we have no hidden for generated tokens and thus do not need to remove anything
        assert isinstance(out[0], torch.Tensor)
        return out

    if output_logits:
        out = out["logits"][0]  # take logits for only generated token
        warnings.warn(
            "Assuming smallLM2 tokenizer to extract relevant logits. Only looking at logits of tokens for answer options"
        )
        # warnings.warn("Assuming two answer options ' A' and ' B'")
        # logger.debug(f"Logits shape: {out.shape}")

        # für sst2 sollte " A" zu label 1 passen, d.h. die Reihenfolge sollte anders sein  -> " B", " A" logits
        # für mnli sollte " A" (entailment) zu label 0, " B" (contra) to 2, " C" to 1 --> " A", " C", " B"

        options_tok_ids = [
            330,  # " A"
            389,  # " B"
            340,  # " C"
            422,  # " D"
            414,  # " E"
            426,  # " F"
            452,  # " G"
            407,  # " H"
            339,  # " I"
        ]
        if n_classes == 2:  # standard sst2
            options_tok_ids = [
                389,  # " B"
                330,  # " A"
            ]
        elif n_classes == 7:  # This is the sst2 memorization setting, where we have 5 additional classes
            options_tok_ids = [
                389,  # " B"
                330,  # " A"
                340,  # " C"
                422,  # " D"
                414,  # " E"
                426,  # " F"
                452,  # " G"
            ]
        elif n_classes == 3:  # standard mnli
            options_tok_ids = [
                330,  # " A"
                340,  # " C"
                389,  # " B"
            ]
        elif n_classes == 8:  # memorization mnli
            options_tok_ids = [
                330,  # " A"
                340,  # " C"
                389,  # " B"
                422,  # " D"
                414,  # " E"
                426,  # " F"
                452,  # " G"
                407,  # " H"
            ]
        else:
            raise ValueError(f"Unexpected number of classes: {n_classes}")

        out = out[:, options_tok_ids[:n_classes]]
        # logger.debug(f"Logits shape: {out.shape}")

        return out


def extract_representations(
    model: Any,
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    dataset: datasets.Dataset,
    prompt_creator: Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]],
    device: str,
    token_pos_to_extract: Optional[int | Literal["mean"]] = None,
    model_type: Literal["sequence-classification", "causal-lm"] = "sequence-classification",
) -> Sequence[torch.Tensor]:
    all_representations = []

    # Batching would be more efficient. But then we need to remove the padding afterwards etc.
    # Representation extraction is not slow enough for me to care.
    prompts = list(map(prompt_creator, dataset))  # type:ignore
    for prompt in tqdm(prompts):
        if model_type == "sequence-classification":
            out = call_sequence_classification_model(model, tokenizer, prompt, device, output_hidden_states=True)
        elif model_type == "causal-lm":
            out = call_causal_lm_model(model, tokenizer, prompt, device, output_hidden_states=True)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        # If we dont need the full representation for all tokens, discard unneeded ones.
        if token_pos_to_extract is not None:
            if isinstance(token_pos_to_extract, int):
                out = tuple((representations[:, token_pos_to_extract, :].unsqueeze(1) for representations in out))
            elif token_pos_to_extract == "mean":
                out = tuple((representations.mean(dim=1).unsqueeze(1) for representations in out))
            else:
                raise ValueError(
                    f"Unknown token_pos_to_extract: {token_pos_to_extract}. Must be integer for specific token or 'mean'"
                )

        out = tuple((representations.to("cpu") for representations in out))
        all_representations.append(out)

    # Combine the list elements (each element corresponds to reps for all layers for one input) into a tuple, where
    # each element corresponds to the representations for all inputs for one layer.
    return to_ntxd_shape(all_representations)


def extract_logits(
    model: Any,
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    dataset: datasets.Dataset,
    prompt_creator: Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]],
    device: str,
    model_type: Literal["sequence-classification", "causal-lm"] = "sequence-classification",
) -> torch.Tensor:
    all_logits = []

    n_classes = len(dataset._info.features["label"].names)
    # Batching would be more efficient. But then we need to remove the padding afterwards etc.
    # Representation extraction is not slow enough for me to care.
    prompts = list(map(prompt_creator, dataset))  # type:ignore
    for prompt in tqdm(prompts):
        if model_type == "sequence-classification":
            out = call_sequence_classification_model(
                model, tokenizer, prompt, device, output_logits=True, output_hidden_states=False
            )
        elif model_type == "causal-lm":
            out = call_causal_lm_model(
                model, tokenizer, prompt, device, output_logits=True, output_hidden_states=False, n_classes=n_classes
            )
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        all_logits.append(out.to("cpu"))

    return torch.cat(all_logits, dim=0)


def to_ntxd_shape(reps: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    concated_reps = []
    n_layers = len(reps[0])
    for layer_idx in range(n_layers):
        concated_reps.append(
            torch.cat(
                [torch.flatten(reps[i][layer_idx], end_dim=-2) for i in range(len(reps))],
                dim=0,
            )
        )
        # logger.debug(f"Layer: {layer_idx}, Shape: {concated_reps[layer_idx].size()}")
    return tuple(concated_reps)


@torch.no_grad()
def get_representations(
    model_path: str,
    model_type: Literal["sequence-classification", "causal-lm"],
    tokenizer_name: str,
    dataset_path: str,
    dataset_config: str | None,
    dataset_local_path: str | None,
    dataset_split: str,
    device: str,
    token_pos: Optional[int | Literal["mean"]] = None,
    shortcut_rate: Optional[float] = None,
    shortcut_seed: Optional[int] = None,
    feature_column: Optional[str] = None,
):
    tokenizer_kwargs = None

    # To avoid loading datasets all the time (which takes considerable time), cache them once we loaded them once.
    dataset_id = "__".join(
        map(
            str,
            [
                dataset_path,
                dataset_config,
                dataset_local_path,
                dataset_split,
                shortcut_rate,
                shortcut_seed,
            ],
        )
    )
    dataset = DATASETS.get(dataset_id, None)
    if dataset is None:
        # This is the first time the dataset gets loaded
        dataset = get_dataset(dataset_path, dataset_config, local_path=dataset_local_path)
        if shortcut_rate is not None and model_type != "causal-lm":
            assert shortcut_seed is not None
            assert feature_column is not None
            logger.info(f"Adding shortcuts with rate {shortcut_rate} and seed {shortcut_seed}")
            label_column = "label"
            shortcut_adder = ShortcutAdder(
                num_labels=len(np.unique(dataset["train"][label_column])),
                p=shortcut_rate,
                feature_column=feature_column,
                label_column=label_column,
                seed=shortcut_seed,
            )
            dataset = dataset.map(shortcut_adder)
            feature_column = shortcut_adder.new_feature_column
            tokenizer_kwargs = {"additional_special_tokens": shortcut_adder.new_tokens}

        dataset = dataset[dataset_split]
        DATASETS[dataset_id] = dataset  # add processed dataset to cache
    prompt_creator = get_prompt_creator(dataset_path, dataset_config, feature_column)

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    tokenizer = get_tokenizer(tokenizer_name, **tokenizer_kwargs)

    with torch.device(device):
        model = get_model(model_path, model_type)
    reps = extract_representations(
        model,
        tokenizer,
        dataset,
        prompt_creator,
        device,
        token_pos_to_extract=token_pos,
        model_type=model_type,
    )
    logger.debug(f"Shape of representations: {[rep.shape for rep in reps]}")
    return reps


# TODO: this is almost exactly duplicated from get_representations. Make this easier to maintain
@torch.no_grad()
def get_logits(
    model_path: str,
    model_type: Literal["sequence-classification", "causal-lm"],
    tokenizer_name: str,
    dataset_path: str,
    dataset_config: str | None,
    dataset_local_path: str | None,
    dataset_split: str,
    device: str,
    shortcut_rate: Optional[float] = None,
    shortcut_seed: Optional[int] = None,
    feature_column: Optional[str] = None,
):
    tokenizer_kwargs = None

    # To avoid loading datasets all the time (which takes considerable time), cache them once we loaded them once.
    dataset_id = "__".join(
        map(
            str,
            [
                dataset_path,
                dataset_config,
                dataset_local_path,
                dataset_split,
                shortcut_rate,
                shortcut_seed,
            ],
        )
    )
    dataset = DATASETS.get(dataset_id, None)
    if dataset is None:
        # This is the first time the dataset gets loaded
        dataset = get_dataset(dataset_path, dataset_config, local_path=dataset_local_path)
        if shortcut_rate is not None and model_type != "causal-lm":
            assert shortcut_seed is not None
            assert feature_column is not None
            logger.info(f"Adding shortcuts with rate {shortcut_rate} and seed {shortcut_seed}")
            label_column = "label"
            shortcut_adder = ShortcutAdder(
                num_labels=len(np.unique(dataset["train"][label_column])),
                p=shortcut_rate,
                feature_column=feature_column,
                label_column=label_column,
                seed=shortcut_seed,
            )
            dataset = dataset.map(shortcut_adder)
            feature_column = shortcut_adder.new_feature_column
            tokenizer_kwargs = {"additional_special_tokens": shortcut_adder.new_tokens}

        dataset = dataset[dataset_split]
        DATASETS[dataset_id] = dataset  # add processed dataset to cache
    prompt_creator = get_prompt_creator(dataset_path, dataset_config, feature_column)

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    tokenizer = get_tokenizer(tokenizer_name, **tokenizer_kwargs)

    with torch.device(device):
        model = get_model(model_path, model_type)
    logits = extract_logits(
        model,
        tokenizer,
        dataset,
        prompt_creator,
        device,
        model_type=model_type,
    )
    logger.debug(f"Shape of logits: {logits.shape}")
    return logits
