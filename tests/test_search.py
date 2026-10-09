"""Regression checks for grid fusion and search termination, without model downloads."""

import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

import search
from utils import general, semba


class SearchTests(unittest.TestCase):
    def test_batched_fusion_matches_scalar_rule(self):
        rng = np.random.default_rng(42)
        beliefs = rng.random((20, 80)) + 1
        scores = rng.random(80)
        expected = np.array(
            [
                state
                * (1 + scores / sum(scores * state))
                / (1 + min(scores) / sum(scores * state))
                for state in beliefs
            ]
        )
        original = beliefs.copy()
        np.testing.assert_allclose(semba.fusion_model(beliefs, scores), expected)
        np.testing.assert_array_equal(beliefs, original)
        np.testing.assert_array_equal(semba.fusion_model(beliefs, scores * 0), beliefs)

    def test_kaplan_uniform_scores_leave_beliefs_unchanged(self):
        beliefs = np.array([1.0, 2.0, 4.0])
        np.testing.assert_array_equal(semba.kaplan(beliefs, [0.5] * 3), beliefs)

    def test_kaplan_known_update_and_score_scaling(self):
        # Equation (1): Z = 7 and m = 2 give [1, 2] * [10/9, 9/9].
        beliefs = np.array([1.0, 2.0])
        scores = np.array([3.0, 2.0])
        expected = [10.0 / 9.0, 2.0]
        np.testing.assert_allclose(semba.kaplan(beliefs, scores), expected)
        np.testing.assert_allclose(semba.kaplan(beliefs, scores * 10), expected)

    def test_kaplan_grid_with_per_cell_and_zero_observations(self):
        beliefs = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        scores = np.array([[[3.0, 2.0], [0.0, 0.0]]])
        with np.errstate(divide="raise", invalid="raise"):
            actual = semba.kaplan(beliefs, scores)
        np.testing.assert_allclose(actual, [[[10.0 / 9.0, 2.0], [3.0, 4.0]]])

    def test_overlap_matches_existing_inclusive_iou(self):
        rng = np.random.default_rng(42)
        shape, height, width = (20, 32), 1050, 1680
        boxes = [[0, 0, 52.5, 52.5], [-10, -10, 0, 0], [1680, 1050, 1700, 1100]]
        for _ in range(100):
            start = rng.uniform(-200, 1800, size=2)
            boxes.append([*start, *(start + rng.uniform(0, 500, size=2))])
        for box in boxes:
            expected = [
                [
                    general.in_cell(y, x, box, shape, height, width)
                    for x in range(shape[1])
                ]
                for y in range(shape[0])
            ]
            actual = search.overlapping_cells(
                box, np.arange(33) * (width / 32), np.arange(21) * (height / 20)
            )
            np.testing.assert_array_equal(actual, expected)

    def test_fixation_selection_and_exhaustion(self):
        attention = np.array([[1.0, 0.5], [0.2, 0.1]])
        inhibited = np.array([[True, False], [False, False]])
        self.assertEqual(search.next_fixation(attention, inhibited), (0, 1))
        self.assertIsNone(search.next_fixation(attention, np.ones((2, 2), dtype=bool)))
        self.assertEqual(
            search.next_fixation(attention * 0, ~np.eye(2, dtype=bool))
            in [(0, 0), (1, 1)],
            True,
        )

    def test_empty_observations_and_attention(self):
        self.assertEqual(semba.fov_observation_model([], 80).shape, (0, 80))
        beliefs = np.array([[[1.0, 3.0], [2.0, 2.0]]])
        np.testing.assert_allclose(
            semba.attention_map(beliefs, (1, 2), 2), [[0.75, 0.5]]
        )

    def test_search_with_empty_predictions_and_immediate_success(self):
        detector = Mock()
        detector.load_model.return_value = (None, None)
        detector.predict.return_value = [[]] * 4
        for threshold, expected_frames in [(1.0, 7), (0.0, 1)]:
            with self.subTest(threshold=threshold), contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.dict(sys.modules, {"utils.detectors": detector})
                )
                stack.enter_context(patch("search.TERMINATION_THRESH", threshold))
                stack.enter_context(
                    patch(
                        "search.skimage.io.imread",
                        return_value=np.zeros((64, 64, 3), dtype=np.uint8),
                    )
                )
                stack.enter_context(
                    patch("search.utils.create_dir", return_value="/tmp/search-test")
                )
                save = stack.enter_context(patch("search.utils.save_map"))
                gif = stack.enter_context(patch("search.utils.generate_gif"))
                plot = stack.enter_context(patch("search.utils.plot_scanpath"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                search.main([])
                self.assertEqual(save.call_count, expected_frames)
                gif.assert_called_once()
                plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
