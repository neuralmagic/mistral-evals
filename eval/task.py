from typing import Any, Optional

import copy
import dataclasses
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.metrics import Metric
from eval.models import Model


@dataclasses.dataclass
class Interaction:
    """A single round of interaction from a model given a chat completion request."""

    # vLLM compatible chat completion request
    request: dict[str, Any]

    # Reference answer(s).
    reference_answer: str | list[str]

    # Generated answer from model.
    model_answer: Optional[str] = None

    # Computed metrics (filled in after model answers are generated).
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)

    # Extra metadata from dataset (e.g. category).
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


class Eval(ABC):
    """Base class for an eval task."""

    def __init__(self):
        self.interactions: list[Interaction] = []

    @property
    def metric_fns(self) -> list[Metric]:
        """A list of metrics to compute for request-response pairs."""
        raise NotImplementedError

    @abstractmethod
    def _to_interaction(self, row: Any):
        """Converts a row from eval dataset into Interaction object."""
        raise NotImplementedError

    @abstractmethod
    def load_eval(self):
        """Loads dataset and applies transforms to get chat completion requests."""
        raise NotImplementedError

    def get_responses(self, model: Model):
        """Queries model to get responses for each interaction."""
        print(f"Total interactions: {len(self.interactions)}")  # Confirm number of samples

        futures: dict[Future, Interaction] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for idx, interaction in enumerate(self.interactions):
                request = copy.deepcopy(interaction.request)
                print(f"Submitting interaction {idx+1}/{len(self.interactions)}")  # Track request submission
                futures[executor.submit(model, request)] = interaction

            interactions_w_model_ans = []
            for future in tqdm(
                as_completed(futures),
                total=len(self.interactions),
                desc="Querying model",
            ):
                interaction = futures[future]
                try:
                    interaction.model_answer = future.result()
                    print(f"Completed {self.interactions.index(interaction) + 1}: {interaction.model_answer}")
                    interactions_w_model_ans.append(interaction)
                except Exception as e:
                    print(f"Error at {self.interactions.index(interaction) + 1}: {e}")
                    print(f"Failed request: {interaction.request}")
            
            self.interactions = interactions_w_model_ans


    def compute_metrics(self):
        """Computes metrics for each interaction."""
        print(f"Computing metrics for {len(self.interactions)} interactions")  

        for interaction in tqdm(self.interactions):
            if interaction.model_answer is None:
                print(f"Warning: Interaction missing model_answer! {interaction}")
                # Set a default score for all metrics since the model did not answer
                interaction.metrics = {metric.name: 0.0 for metric in self.metric_fns}  
            else:
                for metric in self.metric_fns:
                    try:
                        interaction.metrics[metric.name] = metric.score(
                            interaction.model_answer, interaction.reference_answer
                        )
                    except Exception as e:
                        print(f"Metric computation error for {metric.name}: {e}")
                        print(f"Interaction data: {interaction}")


    def aggregate_metrics(self) -> dict[str, float]:
        """Aggregates metrics across all interactions."""
        overall_metrics: dict[str, float] = {}

        for metric in self.metric_fns:
            scores = [interaction.metrics.get(metric.name, float('nan')) for interaction in self.interactions]
            if any(np.isnan(scores)):  # Check for NaNs
                print(f"Warning: NaN detected in {metric.name} scores!")
            
            overall_metrics[metric.name] = np.nanmean(scores)  # Ignore NaNs

        print(f"Final aggregated metrics: {overall_metrics}")
        return overall_metrics



class HuggingFaceEval(Eval):
    """Evals hosted on hugging face for which datasets.load_dataset can be used."""

    dataset_name: str
    dataset_split: str

    def get_dataset(self):
        return load_dataset(self.dataset_name)[self.dataset_split]

    def load_eval(self):
        """Loads dataset and applies transforms to get chat completion requests."""
        dataset = self.get_dataset()
        print(f"Dataset size: {len(dataset)}")  # Check dataset size

        for idx, row in enumerate(tqdm(dataset, desc=f"Loading {self.dataset_name} [{self.dataset_split}]")):
            try:
                interaction = self._to_interaction(row)
                self.interactions.append(interaction)
                if idx % 1000 == 0:  # Print every 1000 samples
                    print(f"Loaded {idx+1} interactions successfully.")
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                print(f"Row causing issue: {row}")  # Print problematic row
                break  # Stop if there's an issue

