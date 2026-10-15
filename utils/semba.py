"""Semantic observation and belief fusion for SemBA."""

# =============================================================================
# 1. Array operations
# =============================================================================

import numpy as np


# =============================================================================
# 2. Semantic evidence fusion
# =============================================================================

def kaplan(belief, scores):
    """Apply Kaplan's rule from the MS-Fovea paper, equation (1).

    For each class k, with Z = sum(belief * scores) and m = min(scores):
        belief_new[k] = belief[k] * (1 + scores[k] / Z) / (1 + m / Z).

    The final axis represents classes; leading dimensions may represent cells.
    Beliefs are positive Dirichlet parameters and scores are nonnegative.
    All-zero observations leave beliefs unchanged (the formula is undefined
    when Z is zero).

    Returns a new array with the broadcast belief shape; neither input is
    modified. A shared score vector can update an entire batch of grid cells.
    """

    belief = np.asarray(belief)
    scores = np.asarray(scores)

    # Retaining the class axis lets a single score vector broadcast over cells.
    weighted_sum = np.sum(scores * belief, axis=-1, keepdims=True)
    min_score = np.min(scores, axis=-1, keepdims=True)
    denominator = weighted_sum + min_score

    # (1 + scores / Z) / (1 + m / Z) = 1 + (scores - m) / (Z + m).
    # This form avoids dividing by Z and preserves beliefs for uniform scores.
    # A zero denominator produces no update for an all-zero observation.
    update = np.divide(
        scores - min_score,
        denominator,
        out=np.zeros_like(belief, dtype=float),
        where=denominator != 0,
    )
    return belief * (1 + update)


def fusion_model(state, scores):
    """Return updated beliefs without modifying the input state.

    state stores positive class beliefs on its final axis. scores may be one
    shared class vector or compatible per-cell observations. This wrapper
    keeps the search loop independent of the selected fusion rule.
    """

    return kaplan(state, scores)


# =============================================================================
# 3. Adapt detector rows to class observations
# =============================================================================

def fov_observation_model(data, total_classes):
    """Extract an (observations, classes) array of foreground scores.

    Detection rows contain four box coordinates, class scores, then a label.
    Empty detections return an empty array with the same class-axis convention.

    total_classes determines the width of the score slice. For nonempty
    input, data must be a two-dimensional array with the shared detection
    layout produced by detectors.predict.
    """

    data = np.asarray(data)
    if data.size == 0:
        return np.empty((0, total_classes))

    # Exclude both the leading box coordinates and the trailing predicted label.
    return data[:, 4 : total_classes + 4]


# =============================================================================
# 4. Project the belief grid onto the requested target class
# =============================================================================

def attention_map(map, cells, class_id):
    """Return target probabilities for a one-based foreground class ID.

    map has shape (rows, columns, classes); cells gives (rows, columns).
    class_id uses the foreground IDs from CLASS_NAMES, starting at one.
    Returns a (rows, columns) array by dividing target beliefs by each cell
    class total. Input beliefs are expected to have positive totals.
    """

    # Validate grid shape and class indexing before normalizing each cell.
    beliefs = np.asarray(map)
    if beliefs.shape[:2] != tuple(cells):
        raise ValueError(
            "The belief grid does not match the requested cell dimensions."
        )
    if not 1 <= class_id <= beliefs.shape[-1]:
        raise ValueError("The target class ID must identify a foreground class.")

    # Convert the one-based target ID to the zero-based foreground belief axis.
    return beliefs[..., class_id - 1] / beliefs.sum(axis=-1)
