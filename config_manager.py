import yaml
import typing
import pathlib
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigException(Exception):
    """Base exception class for configuration-related errors."""
    pass

class InvalidConfigError(ConfigException):
    """Raised when the configuration is invalid."""
    pass

class ConfigManager:
    """
    Configuration management for model parameters, scoring weights, and pruning settings.

    Attributes:
        config_path (Path): Path to the configuration file.
        config (Dict[str, Any]): Loaded configuration dictionary.
    """

    def __init__(self, config_path: Path):
        """
        Initializes the ConfigManager instance.

        Args:
            config_path (Path): Path to the configuration file.

        Raises:
            InvalidConfigError: If the configuration file is invalid.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the specified file path.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.

        Raises:
            InvalidConfigError: If the configuration file is invalid.
        """
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
                if not config:
                    raise InvalidConfigError("Configuration file is empty.")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to load configuration: {e}")
            raise InvalidConfigError("Invalid configuration file.") from e

    def validate_config(self) -> None:
        """
        Validates the loaded configuration.

        Raises:
            InvalidConfigError: If the configuration is invalid.
        """
        required_keys = ["model", "scoring", "pruning"]
        for key in required_keys:
            if key not in self.config:
                raise InvalidConfigError(f"Missing required key: {key}")

        model_params = self.config["model"]
        if not isinstance(model_params, dict):
            raise InvalidConfigError("Model parameters must be a dictionary.")

        scoring_params = self.config["scoring"]
        if not isinstance(scoring_params, dict):
            raise InvalidConfigError("Scoring parameters must be a dictionary.")

        pruning_params = self.config["pruning"]
        if not isinstance(pruning_params, dict):
            raise InvalidConfigError("Pruning parameters must be a dictionary.")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Returns the model parameters from the configuration.

        Returns:
            Dict[str, Any]: Model parameters dictionary.
        """
        return self.config["model"]

    def get_scoring_params(self) -> Dict[str, Any]:
        """
        Returns the scoring parameters from the configuration.

        Returns:
            Dict[str, Any]: Scoring parameters dictionary.
        """
        return self.config["scoring"]

    def get_pruning_params(self) -> Dict[str, Any]:
        """
        Returns the pruning parameters from the configuration.

        Returns:
            Dict[str, Any]: Pruning parameters dictionary.
        """
        return self.config["pruning"]

class ModelParams:
    """
    Model parameters container.

    Attributes:
        num_layers (int): Number of layers in the model.
        hidden_size (int): Hidden size of the model.
        dropout (float): Dropout rate of the model.
    """

    def __init__(self, num_layers: int, hidden_size: int, dropout: float):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

class ScoringParams:
    """
    Scoring parameters container.

    Attributes:
        weights (Dict[str, float]): Weights for scoring.
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

class PruningParams:
    """
    Pruning parameters container.

    Attributes:
        threshold (float): Pruning threshold.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

def main():
    config_path = Path("config.yaml")
    config_manager = ConfigManager(config_path)
    config_manager.validate_config()
    model_params = config_manager.get_model_params()
    scoring_params = config_manager.get_scoring_params()
    pruning_params = config_manager.get_pruning_params()

    logger.info("Model Parameters:")
    logger.info(model_params)
    logger.info("Scoring Parameters:")
    logger.info(scoring_params)
    logger.info("Pruning Parameters:")
    logger.info(pruning_params)

if __name__ == "__main__":
    main()