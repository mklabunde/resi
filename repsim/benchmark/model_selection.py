from collections.abc import Sequence

from repsim.benchmark.registry import TrainedModel


def _group_models(
    models: list[TrainedModel],
    grouping_keys: list[str] | None = None,
) -> list[list[TrainedModel]]:
    """
    Group models based on the differentiation_keys.
    """
    if grouping_keys is None:
        return [models]

    # We find groups that share the same values for the keys chosen to split by.
    # This is done like so:
    # We create a dictionary of list of trained models.
    # The keys of the dict are tuples of the differentation_keys values.
    # By doing so models having the same differentation key values will be grouped together.
    group_dict = {}
    for model in models:
        key = tuple(getattr(model, key) for key in grouping_keys)
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(model)

    grouped_models = list(group_dict.values())
    return grouped_models


def _filter_models(
    models: list[TrainedModel],
    filter_key_vals: dict[str, str | list[str]] | None,
) -> list[TrainedModel]:
    """
    Filter models based on the filter_key_vals.
    """
    if filter_key_vals is None:
        return models

    filtered_models = []

    for model in models:
        matches = True
        for key, val in filter_key_vals.items():
            if not matches:
                break  #
            model_attr_val = getattr(model, key)
            if isinstance(val, Sequence):
                if model_attr_val not in val:
                    matches = False
            else:
                if model_attr_val != val:
                    matches = False
        if matches:
            filtered_models.append(model)
    return filtered_models


def _separate_models_by_keys(
    models: list[TrainedModel], separation_keys: list[str] | None
) -> list[list[TrainedModel]]:
    """
    Separate models that should not be compared to each other.
    Basically identical to the `group_by` key grouping, just a step before!
    """
    if separation_keys is None:
        return [models]
    # Create dict key tuples, creating bins of models sharing the same separation_keys values.
    separate_models = {}
    for model in models:
        key = tuple(getattr(model, key) for key in separation_keys)
        if key not in separate_models:
            separate_models[key] = []
        separate_models[key].append(model)
    return list(separate_models.values())


def get_grouped_models(
    models: list[TrainedModel],
    filter_key_vals: dict[str, str | list[str]] | None,
    separation_keys: list[str] | None = None,
    grouping_keys: list[str] | None = None,
) -> list[list[Sequence[TrainedModel]]]:
    """
    Get a dictionary of models grouped by the values of the filter_key_vals.
    """
    # --------------- Remove unwanted Trained Models from the list --------------- #
    filtered_models = _filter_models(models, filter_key_vals)
    # ---------- Separate models that should not be compared to each other ---------- #
    separated_models: list[list[TrainedModel]] = _separate_models_by_keys(filtered_models, separation_keys)
    # ---------- Group them into groups that get compared to each other ---------- #
    all_grouped_models = []
    for sm in separated_models:
        all_grouped_models.append(_group_models(sm, grouping_keys))
    return all_grouped_models
