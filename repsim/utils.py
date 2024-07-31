import json
import os
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
import torch
from graphs.get_reps import get_gnn_output
from graphs.get_reps import get_graph_model_layer_count
from graphs.get_reps import get_graph_representations
from loguru import logger
from repsim.benchmark.paths import CACHE_PATH
from repsim.benchmark.paths import NLP_MODEL_PATH
from repsim.benchmark.types_globals import DOMAIN_TYPE
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_DOMAIN
from repsim.benchmark.types_globals import NLP_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import NLP_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import VISION_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import VISION_DATASET_TRAINED_ON
from repsim.measures.utils import ND_SHAPE
from repsim.measures.utils import SHAPE_TYPE
from vision.util.data_structs import load_json
from vision.util.file_io import get_vision_model_info
from vision.util.find_architectures import get_base_arch
from vision.util.vision_rep_extraction import get_single_layer_vision_representation_on_demand
from vision.util.vision_rep_extraction import get_vision_output_on_demand

if TYPE_CHECKING:
    from vision.arch.abstract_acti_extr import AbsActiExtrArch
    from vision.util import data_structs as ds
else:
    AbsActiExtrArch = None
    ds = None


# reordering classes does not help, because of cyclical dependencies
ModelRepresentations = TypeVar("ModelRepresentations")
Prediction = TypeVar("Prediction")
Accuracy = TypeVar("Accuracy")


@dataclass
class TrainedModel:
    """
    Class that should contain all the infos about the trained models.
    This can be used to filter the models and categorize them into groups.
    """

    domain: DOMAIN_TYPE
    architecture: VISION_ARCHITECTURE_TYPE | NLP_ARCHITECTURE_TYPE | GRAPH_ARCHITECTURE_TYPE
    train_dataset: VISION_DATASET_TRAINED_ON | NLP_DATASET_TRAINED_ON | GRAPH_DATASET_TRAINED_ON
    identifier: SETTING_IDENTIFIER
    seed: int
    additional_kwargs: Optional[dict] = field(
        kw_only=True, default=None
    )  # Maybe one can remove this to make it more general

    def __post_init__(self):
        self.id = self._get_unique_model_identifier()

    def get_representation(self, representation_dataset: Optional[str] = None, **kwargs) -> ModelRepresentations:
        """
        This function should return the representation of the model.
        """
        raise NotImplementedError()

    def get_output(self, representation_dataset: Optional[str] = None, **kwargs) -> Prediction:
        raise NotImplementedError()

    def get_accuracy(self, representation_dataset: Optional[str] = None, **kwargs) -> Accuracy:
        raise NotImplementedError()

    def _get_unique_model_identifier(self) -> str:
        """
        This function should return a unique identifier for the model.
        """
        return f"{self.domain}_{self.architecture}_{self.train_dataset}_{self.identifier}_{self.seed}_{self.additional_kwargs}"


@dataclass
class NLPDataset:
    name: str  # human-readable identifier
    path: str  # huggingface hub path, e.g., sst2. Will be ignored if `local_path` is given
    config: Optional[str] = None  # huggingface hub dataset config, e.g., mnli for glue.
    local_path: Optional[str] = (
        None  # path to a local dataset directory. If given, the dataset will be loaded from here.
    )
    split: str = "train"  # part of the dataset that was/will be used
    feature_column: Optional[str] = None
    label_column: str = "label"

    # Information about shortcuts
    shortcut_rate: Optional[float] = None
    shortcut_seed: Optional[int] = None

    # Information about changed labels
    memorization_rate: Optional[float] = None
    memorization_n_new_labels: Optional[int] = None
    memorization_seed: Optional[int] = None

    # Information about augmentation
    augmentation_type: Optional[str] = None
    augmentation_rate: Optional[float] = None

    def get_id(self) -> str:
        return "__".join(
            map(
                str,
                [
                    self.path,
                    self.config,
                    self.local_path,
                    self.shortcut_rate,
                    self.shortcut_seed,
                    self.memorization_rate,
                    self.memorization_n_new_labels,
                    self.memorization_seed,
                    self.augmentation_type,
                    self.augmentation_rate,
                ],
            )
        )


@dataclass
class MNLI(NLPDataset):
    path: str = "glue"
    config: str = "mnli"
    feature_column: str = "premise"


@dataclass
class SST2(NLPDataset):
    path: str = "sst2"
    feature_column: str = "sentence"


@dataclass(kw_only=True)
class NLPModel(TrainedModel):
    domain: DOMAIN_TYPE = "NLP"
    architecture: NLP_ARCHITECTURE_TYPE = (
        "BERT-L"  # should actually be Bert-base, but not changing this as it would require changes in the results dataframes (or recomputing them)
    )
    path: str
    tokenizer_name: str
    train_dataset: Literal[
        "sst2", "sst2_sc_rate0558", "sst2_sc_rate0668", "sst2_sc_rate0779", "sst2_sc_rate0889", "sst2_sc_rate10"
    ]
    model_type: Literal["sequence-classification"] = "sequence-classification"
    token_pos: Optional[int] = (
        None  # Index of the token relevant for classification. If set, only the representation of this token will be extracted.
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset_obj: NLPDataset = field(init=False)
    eval_results: None | dict[str, float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.domain != "NLP":
            raise ValueError("This class should only be used for NLP models with huggingface.")

        from repsim.benchmark.registry import NLP_REPRESENTATION_DATASETS, NLP_TRAIN_DATASETS

        self.NLP_REPRESENTATION_DATASETS = NLP_REPRESENTATION_DATASETS
        self.train_dataset_obj = NLP_TRAIN_DATASETS[self.train_dataset]

    @property
    def n_layers(self):
        arch_to_layers = {"BERT-L": 13}
        return arch_to_layers[self.architecture]

    def _check_repsim_dataset_exists(self, representation_dataset_id: str) -> None:
        if representation_dataset_id not in self.NLP_REPRESENTATION_DATASETS.keys():
            raise ValueError(
                f"Dataset must be one of {list(self.NLP_REPRESENTATION_DATASETS.keys())}, but is {representation_dataset_id}"
            )

    def get_representation(
        self, representation_dataset_id: str, compute_on_demand: bool = True
    ) -> ModelRepresentations:
        self._check_repsim_dataset_exists(representation_dataset_id)
        representation_dataset = self.NLP_REPRESENTATION_DATASETS[representation_dataset_id]

        if compute_on_demand:
            slrs = tuple([SingleLayerNLPRepresentation(layer_id=i, _shape="nd") for i in range(self.n_layers)])
        else:
            import repsim.nlp

            reps = repsim.nlp.get_representations(
                self.path,
                self.model_type,
                self.tokenizer_name,
                representation_dataset.path,
                representation_dataset.config,
                representation_dataset.local_path,
                representation_dataset.split,
                self.device,
                self.token_pos,
                shortcut_rate=representation_dataset.shortcut_rate,
                shortcut_seed=representation_dataset.shortcut_seed,
                feature_column=representation_dataset.feature_column,
            )
            slrs = tuple(
                [SingleLayerRepresentation(layer_id=i, _representation=r, _shape="nd") for i, r in enumerate(reps)]
            )
        return ModelRepresentations(
            self,
            representation_dataset_id,
            slrs,
        )

    def get_output(self, representation_dataset_id: str, compute_on_demand: bool = True) -> Prediction:
        self._check_repsim_dataset_exists(representation_dataset_id)
        if compute_on_demand:
            output = NLPModelOutput(origin_model=self, _representation_dataset=representation_dataset_id)
        else:
            import repsim.nlp

            representation_dataset = self.NLP_REPRESENTATION_DATASETS[representation_dataset_id]

            logits = repsim.nlp.get_logits(
                self.path,
                self.model_type,
                self.tokenizer_name,
                representation_dataset.path,
                representation_dataset.config,
                representation_dataset.local_path,
                representation_dataset.split,
                self.device,
                shortcut_rate=representation_dataset.shortcut_rate,
                shortcut_seed=representation_dataset.shortcut_seed,
                feature_column=representation_dataset.feature_column,
            )
            output = Prediction(origin_model=self, _representation_dataset=representation_dataset_id, _output=logits)
        return output

    def get_accuracy(self, representation_dataset_id: str, **kwargs) -> Accuracy:
        return NLPModelAccuracy(origin_model=self, _representation_dataset=representation_dataset_id)


class VisionModel(TrainedModel):

    def _get_unique_model_identifier(self) -> str:
        """
        This function should return a unique identifier for the model.
        """
        return f"{self.domain}_{self.architecture}_{self.train_dataset}_{self.identifier}_{self.seed}"

    def get_representation(self, representation_dataset: Optional[str] = None, **kwargs) -> ModelRepresentations:
        """
        This function should return the representation of the model.
        """

        architecture_name = self.architecture
        train_dataset = self.train_dataset
        seed_id = self.seed
        setting_identifier = self.identifier

        model_info: ds.ModelInfo = get_vision_model_info(
            architecture_name=architecture_name,
            dataset=train_dataset,
            seed_id=seed_id,
            setting_identifier=setting_identifier,
        )

        model_type: Type[AbsActiExtrArch] = get_base_arch(model_info.architecture)
        # # ---------- Create the on-demand-callable functions for each layer ---------- #
        all_single_layer_reps = []
        for i in range(model_type.n_hooks):
            all_single_layer_reps.append(SingleLayerVisionRepresentation(i, origin_model=self))
        model_rep = ModelRepresentations(
            origin_model=self,
            representation_dataset=representation_dataset,
            representations=tuple(all_single_layer_reps),
        )
        return model_rep

    def get_output(self, representation_dataset: Optional[str] = None, **kwargs) -> Prediction:
        out = VisionModelOutput(
            origin_model=self,
            cache=True,
            _representation_dataset=representation_dataset,
        )
        return out

    def get_accuracy(self, representation_dataset: Optional[str] = None, **kwargs) -> Accuracy:
        acc = VisionModelAccuracy(
            origin_model=self,
            _representation_dataset=representation_dataset,
        )
        return acc

    def get_vision_model_info(self):
        return get_vision_model_info(
            architecture_name=self.architecture,
            dataset=self.train_dataset,
            seed_id=self.seed,
            setting_identifier=self.identifier,
        )


class GraphModel(TrainedModel):
    domain: DOMAIN_TYPE = GRAPH_DOMAIN
    architecture: GRAPH_ARCHITECTURE_TYPE
    train_dataset: GRAPH_DATASET_TRAINED_ON

    def _get_unique_model_identifier(self) -> str:
        """
        This function should return a unique identifier for the model.
        """
        return f"{self.domain}_{self.architecture}_{self.train_dataset}_{self.identifier}_{self.seed}"

    def get_representation(
        self,
        representation_dataset: Optional[GRAPH_DATASET_TRAINED_ON] = None,
        compute_on_demand: bool = True,
        **kwargs,
    ) -> ModelRepresentations:
        """
        This function should return the representation of the model.
        """
        if representation_dataset is None:
            representation_dataset = self.train_dataset

        if compute_on_demand:
            n_layers = get_graph_model_layer_count(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed=self.seed,  # type:ignore
                setting_identifier=self.identifier,
            )
            all_single_layer_reps = tuple(
                [SingleLayerGraphRepresentation(layer_id=i, _shape="nd") for i in range(n_layers)]
            )
        else:
            plain_reps = get_graph_representations(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed=self.seed,  # type:ignore
                setting_identifier=self.identifier,
            )

            all_single_layer_reps = []
            for layer_id, rep in plain_reps.items():
                all_single_layer_reps.append(
                    SingleLayerRepresentation(_representation=rep, _shape=ND_SHAPE, layer_id=layer_id)
                )

        return ModelRepresentations(
            origin_model=self,
            representation_dataset=representation_dataset,
            representations=tuple(all_single_layer_reps),
        )

    def get_output(self, representation_dataset: Optional[str] = None, **kwargs) -> Prediction:
        return GraphModelOutput(
            origin_model=self,
            cache=True,
            _representation_dataset=representation_dataset,
        )

    def get_accuracy(self, representation_dataset: Optional[str] = None, **kwargs) -> Accuracy:
        return GraphModelAccuracy(
            origin_model=self,
            _representation_dataset=representation_dataset,
        )


@dataclass
class TrainedModelRep(TrainedModel):
    """
    The same as above, just also has the ID of the layer.
    """

    layer_id: int


@dataclass
class BaseModelOutput(ABC):
    @abstractmethod
    def unique_identifier(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def value_attr_name(self) -> str:
        raise NotImplementedError


@dataclass
class SingleLayerRepresentation(BaseModelOutput):
    layer_id: int
    origin_model: TrainedModel | None = None
    cache: bool = False
    _representation: torch.Tensor | np.ndarray | None = None
    _shape: SHAPE_TYPE | None = None
    _representation_dataset: str | None = field(default=None, init=False)

    def value_attr_name(self) -> str:
        return "representation"

    def representation_is_set(self) -> bool:
        return self._representation is not None

    @property
    def representation(self) -> torch.Tensor | np.ndarray:
        """
        Allows omission of setting representations when creating, and creating on demand.
        """
        if self._representation is None:
            assert self._extract_representation is not None, "No extraction function provided."
            unique_id = self.unique_identifier()
            cache_rep_path = os.path.join(CACHE_PATH, unique_id + ".npz")
            if self.cache and os.path.exists(cache_rep_path):
                self._representation = np.load(cache_rep_path)["representation"]
            else:
                self._representation = self._extract_representation()
                if self.cache:
                    np.savez(cache_rep_path, representation=self._representation)
        return self._representation

    @representation.setter
    def representation(self, v: torch.Tensor | np.ndarray | None) -> None:
        """Allow setting the representation as before"""
        self._representation = v

    # If the GNNs also subclass this class, we can make this abstract, but this method is not strictly required, so I
    # will leave it like this to not break GNN code
    # @abstractmethod
    def _extract_representation(self) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    @property
    def shape(self) -> SHAPE_TYPE:
        if self._shape is None:
            if len(self.representation.shape) == 4:
                shape = "nchw"
            elif len(self.representation.shape) == 3:
                shape = "ntd"
            elif len(self.representation.shape) == 2:
                shape = "nd"
            else:
                raise ValueError(f"Unknown shape of representations: {self.representation.shape}")
        else:
            shape = self._shape
        return shape

    @shape.setter
    def shape(self, v: SHAPE_TYPE) -> None:
        """Allow setting the shape as before"""
        self._shape = v

    def unique_identifier(self) -> str:
        """
        Generates a unique identifier for the SingleLayerRepresentation object.

        Returns:
            A string representing the unique identifier.
        """
        assert self.origin_model is not None, "origin_model is not set"
        assert self._representation_dataset is not None, "representation_dataset not set"
        # We could this make this nicer by using the model .id attribute, but this keeps it backwards compatible
        return "__".join(
            [
                self.origin_model.identifier,
                self.origin_model.architecture,
                self.origin_model.train_dataset,
                str(self.origin_model.seed),
                self._representation_dataset,
                str(self.layer_id),
            ]
        )


class SingleLayerVisionRepresentation(SingleLayerRepresentation):
    def _extract_representation(self) -> torch.Tensor | np.ndarray:
        assert self.origin_model is not None
        return get_single_layer_vision_representation_on_demand(
            architecture_name=self.origin_model.architecture,
            train_dataset=self.origin_model.train_dataset,
            seed=self.origin_model.seed,
            setting_identifier=self.origin_model.identifier,
            representation_dataset=self._representation_dataset,
            layer_id=self.layer_id,
        )


class SingleLayerNLPRepresentation(SingleLayerRepresentation):
    def _extract_representation(self) -> torch.Tensor:
        assert isinstance(self.origin_model, NLPModel)
        assert self._representation_dataset is not None

        import repsim.benchmark.registry
        import repsim.nlp

        representation_dataset = repsim.benchmark.registry.NLP_REPRESENTATION_DATASETS[self._representation_dataset]

        all_layer_reps = repsim.nlp.get_representations(
            self.origin_model.path,
            self.origin_model.model_type,
            self.origin_model.tokenizer_name,
            representation_dataset.path,
            representation_dataset.config,
            representation_dataset.local_path,
            representation_dataset.split,
            self.origin_model.device,
            self.origin_model.token_pos,
            shortcut_rate=representation_dataset.shortcut_rate,
            shortcut_seed=representation_dataset.shortcut_seed,
            feature_column=representation_dataset.feature_column,
        )
        return all_layer_reps[self.layer_id]


class SingleLayerGraphRepresentation(SingleLayerRepresentation):
    def _extract_representation(self) -> torch.Tensor:
        assert isinstance(self.origin_model, GraphModel)

        plain_reps = get_graph_representations(
            architecture_name=self.origin_model.architecture,
            train_dataset=self.origin_model.train_dataset,
            seed=self.origin_model.seed,  # type:ignore
            setting_identifier=self.origin_model.identifier,
        )
        return plain_reps[self.layer_id]


@dataclass
class ModelRepresentations:
    origin_model: TrainedModel
    representation_dataset: str
    representations: tuple[SingleLayerRepresentation, ...]  # immutable to maintain ordering

    def __post_init__(self):
        """Automatically adds the ModelRepresentations infos to the SingleLayerRepresentations infos."""
        if self.representations is not None:
            self._set_single_layer_infos()
        else:
            logger.warning(
                """ModelRepresentations has not been set with the necessary information. Make sure to call
                `_set_single_layer_infos` or saving will fail!"""
            )

    def _set_single_layer_infos(self):
        for rep in self.representations:
            rep.origin_model = self.origin_model
            rep._representation_dataset = self.representation_dataset


@dataclass
class Prediction(BaseModelOutput):
    origin_model: TrainedModel | None = None
    cache: bool = False
    _representation_dataset: str | None = None
    _output: torch.Tensor | np.ndarray | None = None

    def value_attr_name(self) -> str:
        return "output"

    @property
    def output(self) -> torch.Tensor | np.ndarray:
        """
        Allows omission of setting outputs when creating, and creating on demand.
        """
        if self._output is None:
            assert self._extract_output is not None, "No extraction function provided."
            unique_id = self.unique_identifier()
            cache_path = os.path.join(CACHE_PATH, unique_id + ".npz")
            if self.cache and os.path.exists(cache_path):
                self._output = np.load(cache_path)["output"]
            else:
                self._output = self._extract_output()
                if self.cache:
                    np.savez(cache_path, output=self._output)
        assert self._output is not None
        return self._output

    @output.setter
    def output(self, v: torch.Tensor | np.ndarray) -> None:
        self._output = v

    @abstractmethod
    def _extract_output(self) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    def unique_identifier(self) -> str:
        """
        Generates a unique identifier for this object.

        Returns:
            A string representing the unique identifier.
        """
        assert self.origin_model is not None, "origin_model is not set"
        assert self._representation_dataset is not None, "representation_dataset not set"
        return "__".join([self.origin_model.id, self._representation_dataset, "output"])


class NLPModelOutput(Prediction):
    def _extract_output(self) -> torch.Tensor | np.ndarray:
        assert isinstance(self.origin_model, NLPModel)
        assert self._representation_dataset is not None

        import repsim.nlp

        representation_dataset = self.origin_model.NLP_REPRESENTATION_DATASETS[self._representation_dataset]

        logits = repsim.nlp.get_logits(
            self.origin_model.path,
            self.origin_model.model_type,
            self.origin_model.tokenizer_name,
            representation_dataset.path,
            representation_dataset.config,
            representation_dataset.local_path,
            representation_dataset.split,
            self.origin_model.device,
            shortcut_rate=representation_dataset.shortcut_rate,
            shortcut_seed=representation_dataset.shortcut_seed,
            feature_column=representation_dataset.feature_column,
        )
        return logits


class VisionModelOutput(Prediction):
    def _extract_output(self) -> torch.Tensor | np.ndarray:
        assert self.origin_model is not None
        logits = get_vision_output_on_demand(
            architecture_name=self.origin_model.architecture,
            train_dataset=self.origin_model.train_dataset,
            seed=self.origin_model.seed,
            setting_identifier=self.origin_model.identifier,
            representation_dataset=self._representation_dataset,
        )
        return logits


class GraphModelOutput(Prediction):
    origin_model: GraphModel | None = None

    def _extract_output(self) -> torch.Tensor | np.ndarray:
        assert self.origin_model is not None
        return get_gnn_output(
            architecture_name=self.origin_model.architecture,
            train_dataset=self.origin_model.train_dataset,
            seed=self.origin_model.seed,  # type:ignore
            setting_identifier=self.origin_model.identifier,
        )


@dataclass
class Accuracy(BaseModelOutput):
    origin_model: TrainedModel | None = None
    cache: bool = False
    _representation_dataset: str | None = None
    _acc: float | None = None

    def value_attr_name(self) -> str:
        return "accuracy"

    @property
    def accuracy(self) -> float:
        """
        Allows omission of setting outputs when creating, and creating on demand.
        """
        if self._acc is None:
            assert self._extract_output is not None, "No extraction function provided."
            unique_id = self.unique_identifier()
            cache_path = os.path.join(CACHE_PATH, unique_id + ".npz")
            if self.cache and os.path.exists(cache_path):
                self._acc = np.load(cache_path)["output"]
            else:
                self._acc = self._extract_output()
                if self.cache:
                    np.savez(cache_path, output=self._acc)
        assert self._acc is not None
        return self._acc

    @accuracy.setter
    def accuracy(self, v: float) -> None:
        self._acc = v

    @abstractmethod
    def _extract_output(self) -> float:
        raise NotImplementedError

    def unique_identifier(self) -> str:
        """
        Generates a unique identifier for this object.

        Returns:
            A string representing the unique identifier.
        """
        assert self.origin_model is not None, "origin_model is not set"
        assert self._representation_dataset is not None, "representation_dataset not set"
        return "__".join([self.origin_model.id, self._representation_dataset, "accuracy"])


class NLPModelAccuracy(Accuracy):
    def _extract_output(self) -> float:
        assert isinstance(self.origin_model, NLPModel)
        assert self._representation_dataset is not None

        from repsim.benchmark.registry import NLP_REPRESENTATION_DATASETS

        split = NLP_REPRESENTATION_DATASETS[self._representation_dataset].split

        with (NLP_MODEL_PATH / "eval_results.json").open("r") as f:
            try:
                eval_results = json.load(f)[self.origin_model.id]
                return eval_results[self._representation_dataset][split]["accuracy"]
            except KeyError as e:
                logger.error(f"Unable to load eval results of {self.origin_model.id}: {e}")
                return np.nan


class VisionModelAccuracy(Accuracy):
    def _extract_output(self) -> float:

        assert self.origin_model is not None
        og_model = self.origin_model
        info = get_vision_model_info(
            og_model.train_dataset, og_model.architecture, og_model.seed, og_model.identifier
        ).path_output_json
        assert info.exists(), "Output json does not exist -- Can't load accuracy"
        acc = load_json(info)["test"]["accuracy"]
        return acc


class GraphModelAccuracy(Accuracy):
    origin_model: GraphModel | None = None

    def _extract_output(self) -> float:
        assert self.origin_model is not None
        _, acc = get_gnn_output(
            architecture_name=self.origin_model.architecture,
            train_dataset=self.origin_model.train_dataset,
            seed=self.origin_model.seed,  # type:ignore
            setting_identifier=self.origin_model.identifier,
            return_accuracy=True,
        )
        return acc


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")


@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield
