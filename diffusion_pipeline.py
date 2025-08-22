import diffusers
import torch
import transformers
import accelerate
import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

class DiffusionPipelineWrapper:
    """
    Wrapper class for various diffusion models with intermediate prediction extraction capabilities.

    Parameters:
    - model_name_or_path (str): Name or path of the diffusion model to load.
    - device (str, optional): Device to use for computations ('cpu' or 'cuda'). Default is 'cuda' if available, otherwise 'cpu'.
    - num_inference_steps (int, optional): Number of inference steps to use. Default is 50.
    - progressive_pruning (bool, optional): Whether to use progressive pruning strategy. Default is True.
    - velocity_threshold (float, optional): Velocity threshold for progressive pruning. Default is 0.5.
    - **kwargs: Additional keyword arguments to pass to the underlying model.

    Attributes:
    - model (diffusers.models.BaseModel): The loaded diffusion model.
    - device (torch.device): Device used for computations.
    - num_inference_steps (int): Number of inference steps.
    - progressive_pruning (bool): Whether progressive pruning is enabled.
    - velocity_threshold (float): Velocity threshold for progressive pruning.
    """

    def __init__(self, model_name_or_path: str, device: str = None, num_inference_steps: int = 50, progressive_pruning: bool = True, velocity_threshold: float = 0.5, **kwargs):
        self.model = diffusers.load_model(model_name_or_path, device=device, **kwargs)
        self.device = self.model.device
        self.num_inference_steps = num_inference_steps
        self.progressive_pruning = progressive_pruning
        self.velocity_threshold = velocity_threshold

        # Additional configuration and setup
        self.model.eval()  # Ensure model is in evaluation mode

    def generate_intermediate_predictions(self, x_t: torch.Tensor, num_intermediate: int = 4, return_all: bool = False) -> torch.Tensor:
        """
        Generate intermediate predictions for a given noisy input x_t.

        Parameters:
        - x_t (torch.Tensor): Noisy input tensor of shape (batch_size, channels, height, width).
        - num_intermediate (int, optional): Number of intermediate predictions to generate. Default is 4.
        - return_all (bool, optional): Whether to return all intermediate predictions or just the final one. Default is False.

        Returns:
        - torch.Tensor: Tensor of intermediate predictions of shape (batch_size, num_intermediate, channels, height, width).
                      If return_all is False, the shape is (batch_size, channels, height, width).
        """
        batch_size, _, height, width = x_t.shape
        intermediate_preds = []
        alphas = np.linspace(1, 0, num_intermediate + 2)[1:-1]  # [1, 0.8, 0.6, ..., 0.2]

        for alpha in alphas:
            x_t_recon = self.model.ddpm_sample(x_t, num_inference_steps=self.num_inference_steps, clip_alphas=[alpha], progressive=self.progressive_pruning, velocity_threshold=self.velocity_threshold)
            intermediate_preds.append(x_t_recon)

        intermediate_preds = torch.stack(intermediate_preds, dim=1)

        if not return_all:
            intermediate_preds = intermediate_preds[:, -1]

        return intermediate_preds

    def extract_x0_prediction(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Extract the x0 prediction (clean image) from a given noisy input x_t.

        Parameters:
        - x_t (torch.Tensor): Noisy input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Tensor of x0 predictions of shape (batch_size, channels, height, width).
        """
        return self.generate_intermediate_predictions(x_t, num_intermediate=1, return_all=True)[:, 0]

    def get_model_config(self) -> dict:
        """
        Get the configuration of the loaded diffusion model.

        Returns:
        - dict: Dictionary containing the model configuration.
        """
        return self.model.config.to_dict()

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load diffusion model wrapper
    model_wrapper = DiffusionPipelineWrapper("stable_diffusion_v1_5_original", num_inference_steps=50, velocity_threshold=0.6)

    # Generate some random noisy input
    batch_size = 4
    channels = 3
    height, width = 256, 256
    x_t = torch.rand(batch_size, channels, height, width, device=model_wrapper.device)

    # Generate intermediate predictions
    intermediate_preds = model_wrapper.generate_intermediate_predictions(x_t, num_intermediate=4, return_all=True)
    logger.info(f"Intermediate predictions shape: {intermediate_preds.shape}")

    # Extract x0 prediction
    x0_pred = model_wrapper.extract_x0_prediction(x_t)
    logger.info(f"x0 prediction shape: {x0_pred.shape}")

    # Get model configuration
    model_config = model_wrapper.get_model_config()
    logger.info(f"Model configuration: {model_config}")