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

    # def get_responses(self, model: Model):
    #     """Queries model to get responses for each interaction."""

    #     futures: dict[Future, Interaction] = {}
    #     with ThreadPoolExecutor(max_workers=8) as executor:
    #         for interaction in self.interactions:
    #             request = copy.deepcopy(interaction.request)
    #             futures[executor.submit(model, request)] = interaction

    #         interactions_w_model_ans = []
    #         for future in tqdm(
    #             as_completed(futures),
    #             total=len(self.interactions),
    #             desc="Querying model",
    #         ):
    #             interaction = futures[future]
    #             interaction.model_answer = future.result()
    #             interactions_w_model_ans.append(interaction)
    #         self.interactions = interactions_w_model_ans
    
    def get_responses(self, model: Model):
        """Queries model for a limited range of interactions."""
        if not self.interactions:
            print("No interactions loaded. Run `load_eval` first.")
            return

        failure_idx = int(0.67 * len(self.interactions))
        start_idx = max(0, failure_idx - 10)
        end_idx = min(len(self.interactions), failure_idx + 10)
        
        futures: dict[Future, Interaction] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for idx in range(start_idx, end_idx):
                interaction = self.interactions[idx]
                request = copy.deepcopy(interaction.request)
                print(f"Processing interaction {idx + 1}/{len(self.interactions)}")
                futures[executor.submit(model, request)] = interaction

            interactions_w_model_ans = []
            for future in tqdm(
                as_completed(futures),
                total=(end_idx - start_idx),
                desc="Querying model subset",
            ):
                interaction = futures[future]
                try:
                    interaction.model_answer = future.result()
                    print(f"Completed {self.interactions.index(interaction) + 1}: {interaction.model_answer}")
                    interactions_w_model_ans.append(interaction)
                except Exception as e:
                    print(f"Error at {self.interactions.index(interaction) + 1}: {e}")
            
            self.interactions = interactions_w_model_ans


    def compute_metrics(self):
        """Computes metrics for each interaction."""
        for interaction in tqdm(self.interactions):
            for metric in self.metric_fns:
                interaction.metrics[metric.name] = metric.score(
                    interaction.model_answer, interaction.reference_answer
                )

    def aggregate_metrics(self) -> dict[str, float]:
        """Aggregates metrics across all the interactions."""
        overall_metrics: dict[str, float] = {}
        for metric in self.metric_fns:
            overall_metrics[metric.name] = np.mean(
                [interaction.metrics[metric.name] for interaction in self.interactions]
            )  # type: ignore
        return overall_metrics


class HuggingFaceEval(Eval):
    """Evals hosted on hugging face for which datasets.load_dataset can be used."""

    dataset_name: str
    dataset_split: str

    def get_dataset(self):
        return load_dataset(self.dataset_name)[self.dataset_split]

    # def load_eval(self):
    #     """Loads dataset and applies transforms to get chat completion requests."""
    #     for row in tqdm(
    #         self.get_dataset(),
    #         desc=f"Loading {self.dataset_name} [{self.dataset_split}]",
    #     ):
    #         self.interactions.append(self._to_interaction(row))

    def load_eval(self):
        """Loads only a subset of the dataset around the failure point."""
        dataset = self.get_dataset()
        failure_idx = int(0.67 * len(dataset))
        start_idx = max(0, failure_idx - 10)
        end_idx = min(len(dataset), failure_idx + 10)

        for idx, row in enumerate(tqdm(
            dataset.select(range(start_idx, end_idx)),
            desc=f"Loading subset {start_idx} to {end_idx}",
        )):
            try:
                interaction = self._to_interaction(row)
                print(f"Loaded interaction {start_idx + idx}: {interaction.request}")
                self.interactions.append(interaction)
            except Exception as e:
                print(f"Error loading interaction {start_idx + idx}: {e}")

