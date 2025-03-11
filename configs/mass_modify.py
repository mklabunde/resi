#!/usr/bin/env python3
import argparse
from pathlib import Path

from ruamel.yaml import YAML


def modify_yaml_configs(
    directory: str,
    only_eval: bool,
    rerun_nans: bool,
    excluded_measures: list[str] | None,
    included_measures: list[str] | None,
):
    """Modify all YAML configs in the given directory.

    Args:
        directory: Path to directory containing YAML configs
        only_eval: Value to set for only_eval flag
        rerun_nans: Value to set for rerun_nans flag
        excluded_measures: List of measures to exclude, or None to leave unchanged
    """
    yaml = YAML()
    yaml.preserve_quotes = True

    # Find all yaml files in directory
    config_dir = Path(directory)
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    for config_file in yaml_files:
        print(f"Modifying {config_file}")

        # Load existing config
        with open(config_file, "r") as f:
            config = yaml.load(f)

        # Update values
        config["only_eval"] = only_eval
        config["rerun_nans"] = rerun_nans
        if excluded_measures is not None:
            # Remove existing excluded_measures if present
            config.pop("included_measures", None)
            config["excluded_measures"] = excluded_measures

        if included_measures is not None:
            # Remove existing excluded_measures if present
            config.pop("excluded_measures", None)
            config["included_measures"] = included_measures

        # Write back to file
        with open(config_file, "w") as f:
            yaml.dump(config, f)


def main():
    parser = argparse.ArgumentParser(description="Modify YAML config files in bulk")
    parser.add_argument("directory", help="Directory containing YAML configs")
    parser.add_argument("--only-eval", action="store_true", help="Set only_eval to True")
    parser.add_argument("--rerun-nans", action="store_true", help="Set rerun_nans to True")
    parser.add_argument("--excluded-measures", help="Comma-separated list of measures to exclude")
    parser.add_argument("--included-measures", help="Comma-separated list of measures to include")

    args = parser.parse_args()

    excluded_measures = args.excluded_measures.split(",") if args.excluded_measures else None
    included_measures = args.included_measures.split(",") if args.included_measures else None

    modify_yaml_configs(args.directory, args.only_eval, args.rerun_nans, excluded_measures, included_measures)


if __name__ == "__main__":
    main()
