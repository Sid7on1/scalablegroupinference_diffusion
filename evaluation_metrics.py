import torch
import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import logging.config
from typing import List, Dict
from config import Config
from utils import load_config, setup_logging
from data import load_data
from metrics import compute_diversity_score, compute_quality_score, compute_combined_score
from visualization import plot_pareto_frontier
from statistics import bootstrap_confidence_interval

# Set up logging
logging.config.dictConfig(load_config('logging'))
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self, config: Config):
        self.config = config
        self.data = load_data(config.data_path)

    def compute_diversity_score(self, embeddings: torch.Tensor) -> float:
        """
        Compute diversity score using silhouette coefficient.

        Args:
        embeddings (torch.Tensor): Embeddings of generated images.

        Returns:
        float: Diversity score.
        """
        try:
            # Standardize embeddings
            scaler = StandardScaler()
            standardized_embeddings = scaler.fit_transform(embeddings.detach().numpy())

            # Compute silhouette score
            diversity_score = silhouette_score(standardized_embeddings, np.argmax(embeddings, axis=1))

            return diversity_score
        except Exception as e:
            logger.error(f"Error computing diversity score: {e}")
            return None

    def compute_quality_score(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute quality score using cosine similarity.

        Args:
        embeddings (torch.Tensor): Embeddings of generated images.
        labels (torch.Tensor): Labels of generated images.

        Returns:
        float: Quality score.
        """
        try:
            # Compute cosine similarity
            quality_score = torch.cosine_similarity(embeddings, labels).mean().item()

            return quality_score
        except Exception as e:
            logger.error(f"Error computing quality score: {e}")
            return None

    def compute_combined_score(self, diversity_score: float, quality_score: float) -> float:
        """
        Compute combined score as a weighted sum of diversity and quality scores.

        Args:
        diversity_score (float): Diversity score.
        quality_score (float): Quality score.

        Returns:
        float: Combined score.
        """
        try:
            # Compute combined score
            combined_score = (self.config.diversity_weight * diversity_score +
                              self.config.quality_weight * quality_score)

            return combined_score
        except Exception as e:
            logger.error(f"Error computing combined score: {e}")
            return None

    def plot_pareto_frontier(self, diversity_scores: List[float], quality_scores: List[float]) -> None:
        """
        Plot Pareto frontier of diversity and quality scores.

        Args:
        diversity_scores (List[float]): Diversity scores.
        quality_scores (List[float]): Quality scores.
        """
        try:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot Pareto frontier
            sns.scatterplot(x=diversity_scores, y=quality_scores, ax=ax)

            # Set title and labels
            ax.set_title("Pareto Frontier")
            ax.set_xlabel("Diversity Score")
            ax.set_ylabel("Quality Score")

            # Show plot
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting Pareto frontier: {e}")

    def bootstrap_confidence_interval(self, scores: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for a list of scores.

        Args:
        scores (List[float]): List of scores.
        confidence_level (float, optional): Confidence level. Defaults to 0.95.

        Returns:
        Dict[str, float]: Confidence interval.
        """
        try:
            # Compute bootstrap confidence interval
            ci = bootstrap_confidence_interval(scores, confidence_level=confidence_level)

            return ci
        except Exception as e:
            logger.error(f"Error computing bootstrap confidence interval: {e}")
            return None


def main():
    # Load configuration
    config = load_config('evaluation')

    # Set up logging
    setup_logging(config.logging)

    # Create evaluation metrics object
    evaluation_metrics = EvaluationMetrics(config)

    # Compute diversity and quality scores
    diversity_scores = []
    quality_scores = []
    for i in range(len(evaluation_metrics.data)):
        embeddings = evaluation_metrics.data[i]['embeddings']
        labels = evaluation_metrics.data[i]['labels']
        diversity_score = evaluation_metrics.compute_diversity_score(embeddings)
        quality_score = evaluation_metrics.compute_quality_score(embeddings, labels)
        diversity_scores.append(diversity_score)
        quality_scores.append(quality_score)

    # Compute combined score
    combined_scores = []
    for i in range(len(diversity_scores)):
        combined_score = evaluation_metrics.compute_combined_score(diversity_scores[i], quality_scores[i])
        combined_scores.append(combined_score)

    # Plot Pareto frontier
    evaluation_metrics.plot_pareto_frontier(diversity_scores, quality_scores)

    # Compute bootstrap confidence interval
    ci = evaluation_metrics.bootstrap_confidence_interval(combined_scores)
    print(f"Bootstrap confidence interval: {ci}")


if __name__ == "__main__":
    main()