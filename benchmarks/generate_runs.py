import argparse
import yaml
import itertools
import os
from subprocess import run


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def generate_configs(param_grid):
    # Extract top-level parameters
    datasets = param_grid["dataset"]
    featurizers = param_grid["featurizer"]
    models = param_grid["model"]
    cross_validation = param_grid['cross_validation']

    featurizer_kwargs = param_grid.get("featurizer_args", {})
    model_kwargs = param_grid.get("model_args", {})

    # Generate all combinations of dataset, featurizer, and model
    top_combinations = list(itertools.product(datasets, featurizers, models, cross_validation))

    all_configs = []
    # Iterate through all combinations of top-level parameters
    for dataset, featurizer, model, cross_validation in top_combinations:
        # Generate featurizer kwargs for the selected featurizer
        featurizer_config = {}
        if featurizer in featurizer_kwargs:
            for key, values in featurizer_kwargs[featurizer].items():
                featurizer_config[key] = values

        # Generate model kwargs for the selected model
        model_config = {}
        if model in model_kwargs:
            for key, values in model_kwargs[model].items():
                model_config[key] = values

        # Combine the top-level parameters with the specific kwargs
        # Create all combinations of featurizer_kwargs and model_kwargs
        featurizer_combinations = (
            list(itertools.product(*featurizer_config.values()))
            if featurizer_config
            else [()]
        )
        model_combinations = (
            list(itertools.product(*model_config.values())) if model_config else [()]
        )

        # Generate full configuration by combining top-level, featurizer, and model kwargs
        for featurizer_params in featurizer_combinations:
            for model_params in model_combinations:
                config = {
                    "dataset": dataset,
                    "featurizer": featurizer,
                    "featurizer_args": {},
                    "model": model,
                    "model_args": {},
                    "cross_validation": cross_validation,
                }

                # Add featurizer kwargs
                for key, value in zip(featurizer_config.keys(), featurizer_params):
                    config["featurizer_args"][key] = value

                # Add model kwargs
                for key, value in zip(model_config.keys(), model_params):
                    config["model_args"][key] = value

                all_configs.append(config)
    return all_configs


def main(yaml_file, results_dir):
    param_grid = load_yaml(yaml_file)
    configs = generate_configs(param_grid)
    # Create a directory for each config and write the config to a file in that directory
    for i, config in enumerate(configs):
        config_dir = os.path.join(results_dir, f"run_config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.yaml")
        config_file = os.path.abspath(config_file)
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        command = [] # command for job scheduler
        run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configurations for training models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file for generating runs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./",
        help="Directory containing output files",
    )
    args = parser.parse_args()

    config = args.config
    results_dir = args.results_dir
    main(config, results_dir)