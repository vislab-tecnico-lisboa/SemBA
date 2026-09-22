"""Semantic observation and belief fusion for SemBA."""

import numpy as np


def kaplan(belief, scores):
    """Apply Kaplan's rule along the final (class) axis.

    Leading dimensions may represent multiple cells. All-zero observations
    carry no information and leave the beliefs unchanged.
    """
    belief = np.asarray(belief)
    scores = np.asarray(scores)
    weighted_sum = np.sum(scores * belief, axis=-1, keepdims=True)
    denominator = weighted_sum + np.min(scores, axis=-1, keepdims=True)
    update = np.divide(scores, denominator, out=np.zeros_like(belief, dtype=float),
                       where=denominator != 0)
    return belief * (1 + update)


def fusion_model(state, scores):
    """Return updated beliefs without modifying the input state."""
    return kaplan(state, scores)


def fov_observation_model(data, total_classes):
    """Extract class scores, including a correctly shaped empty result."""
    data = np.asarray(data)
    if data.size == 0:
        return np.empty((0, total_classes))
    return data[:, 4:total_classes + 4]


def attention_map(map, cells, class_id):
    """Return target probabilities for a one-based foreground class ID."""
    beliefs = np.asarray(map)
    if beliefs.shape[:2] != tuple(cells):
        raise ValueError('The belief grid does not match the requested cell dimensions.')
    if not 1 <= class_id <= beliefs.shape[-1]:
        raise ValueError('The target class ID must identify a foreground class.')
    return beliefs[..., class_id - 1] / beliefs.sum(axis=-1)
