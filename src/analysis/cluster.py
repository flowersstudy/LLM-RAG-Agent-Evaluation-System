"""
Failure clustering: groups similar failures using embedding similarity.

Uses SentenceTransformer embeddings + DBSCAN to discover systematic
failure patterns without requiring pre-specified cluster count.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from src.core.models import EvaluationResult, FailureCluster, FailureType
from src.utils.logging import get_logger

logger = get_logger()


class FailureClusterer:
    """Clusters failure descriptions into groups of similar failures."""

    def __init__(self, eps: float = 0.3, min_samples: int = 2) -> None:
        self._eps = eps
        self._min_samples = min_samples
        self._model = None

    def _get_model(self):
        """Lazy-load embedding model (uses cached all-MiniLM-L6-v2)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import os
            try:
                # Force offline to avoid HuggingFace network issues
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self._model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Clustering disabled.")
                return None
        return self._model

    def cluster(
        self,
        results: List[EvaluationResult],
        tasks: list,
    ) -> List[FailureCluster]:
        """
        Group failures from results into clusters.

        Returns empty list if fewer than min_samples failures exist.
        """
        # Collect all failures with their descriptions
        failures: List[tuple] = []  # (result, failure_mode, task)
        for r in results:
            task = next((t for t in tasks if t.id == r.task_id), None)
            for fm in r.failure_modes:
                failures.append((r, fm, task))

        if len(failures) < self._min_samples:
            logger.info(f"Too few failures ({len(failures)}) for clustering, need {self._min_samples}")
            return []

        descriptions = [fm.description for _, fm, _ in failures]

        # Embed
        model = self._get_model()
        if model is None:
            logger.warning("Embedding model unavailable, skipping clustering")
            return []
        embeddings = model.encode(descriptions, show_progress_bar=False)

        # DBSCAN clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=self._eps, min_samples=self._min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        clusters: List[FailureCluster] = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            indices = [i for i, l in enumerate(labels) if l == label]
            cluster_failures = [failures[i] for i in indices]
            cluster_descs = [failures[i][1].description for i in indices]

            # Dominant failure type
            type_counts = Counter(failures[i][1].type for i in indices)
            dominant_type = type_counts.most_common(1)[0][0]

            # Generate label from most common words
            all_words = " ".join(d.replace("=", " ").replace("(", " ").replace(")", " ") for d in cluster_descs).lower().split()
            word_counts = Counter(w for w in all_words if len(w) > 3)
            top_words = [w for w, _ in word_counts.most_common(3)]

            # Representative examples
            reps = [failures[i][0].task_id for i in indices[:3]]

            clusters.append(FailureCluster(
                cluster_id=label + 1,
                label=f"Cluster {label + 1}: {' / '.join(top_words)}",
                failure_type=dominant_type,
                size=len(indices),
                representative_examples=reps,
                description="\n".join(set(failures[i][1].description for i in indices)),
            ))

        logger.info(f"Found {len(clusters)} failure clusters (noise: {sum(1 for l in labels if l == -1)})")
        return clusters
