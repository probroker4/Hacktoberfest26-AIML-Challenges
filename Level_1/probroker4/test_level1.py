import math
import os
import tempfile
import unittest

from similarity import (
    cosine_similarity,
    dot,
    load_table,
    magnitude,
    min_max_fit,
    min_max_scale,
    rank,
)
from squashing import relu, sigmoid, sigmoid_naive, tanh


class TestDot(unittest.TestCase):
    def test_known_value(self):
        # 1*4 + 2*5 + 3*6 = 32
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dot([1, 2], [1, 2, 3])

    def test_magnitude_of_3_4_is_5(self):
        self.assertAlmostEqual(magnitude([3, 4]), 5.0)


class TestCosineSimilarity(unittest.TestCase):
    def test_same_direction_is_1(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_perpendicular_is_0(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_direction_is_minus_1(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [-1, -2]), -1.0)

    def test_length_does_not_matter(self):
        # Stretching a vector changes its length, not its direction.
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [10, 20, 30]), 1.0)

    def test_zero_vector_is_0(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)


class TestMinMax(unittest.TestCase):
    def setUp(self):
        self.rows = [[10, 0.2], [20, 0.4], [30, 1.0]]
        self.mins, self.maxs = min_max_fit(self.rows)

    def test_fit_finds_column_min_and_max(self):
        self.assertEqual(self.mins, [10, 0.2])
        self.assertEqual(self.maxs, [30, 1.0])

    def test_min_maps_to_0_and_max_to_1(self):
        self.assertEqual(min_max_scale([10, 0.2], self.mins, self.maxs), [0.0, 0.0])
        self.assertEqual(min_max_scale([30, 1.0], self.mins, self.maxs), [1.0, 1.0])

    def test_midpoint_maps_to_half(self):
        scaled = min_max_scale([20, 0.6], self.mins, self.maxs)
        self.assertAlmostEqual(scaled[0], 0.5)
        self.assertAlmostEqual(scaled[1], 0.5)

    def test_constant_column_does_not_divide_by_zero(self):
        mins, maxs = min_max_fit([[5, 1], [5, 2]])
        self.assertEqual(min_max_scale([5, 2], mins, maxs), [0.0, 1.0])


class TestTableAndRanking(unittest.TestCase):
    def test_load_and_rank_a_small_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "table.csv")
            with open(path, "w") as f:
                f.write("name,x,y\nright,1,0\nup,0,1\ndiagonal,1,1\n")
            labels, rows = load_table(path)

        self.assertEqual(labels, ["right", "up", "diagonal"])
        self.assertEqual(rows, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        ranking = rank([2, 0], labels, rows)
        self.assertEqual([label for label, _ in ranking], ["right", "diagonal", "up"])


class TestSigmoid(unittest.TestCase):
    def test_zero_gives_half(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_matches_naive_for_ordinary_inputs(self):
        for x in [-5, -1, 0.5, 3]:
            self.assertAlmostEqual(sigmoid(x), sigmoid_naive(x))

    def test_naive_breaks_on_large_negative_input(self):
        with self.assertRaises(OverflowError):
            sigmoid_naive(-1000)

    def test_fixed_version_survives_extreme_inputs(self):
        self.assertEqual(sigmoid(-1000), 0.0)
        self.assertEqual(sigmoid(1000), 1.0)

    def test_output_stays_between_0_and_1(self):
        for x in [-50, -2, 0, 2, 50]:
            self.assertTrue(0.0 <= sigmoid(x) <= 1.0)


class TestRelu(unittest.TestCase):
    def test_negative_becomes_0(self):
        self.assertEqual(relu(-3.5), 0.0)

    def test_zero_stays_0(self):
        self.assertEqual(relu(0), 0.0)

    def test_positive_is_unchanged(self):
        self.assertEqual(relu(2.5), 2.5)


class TestTanh(unittest.TestCase):
    def test_matches_math_tanh(self):
        for x in [-10, -1, -0.001, 0, 0.001, 1, 10]:
            self.assertAlmostEqual(tanh(x), math.tanh(x), delta=1e-12)

    def test_survives_extreme_inputs(self):
        self.assertEqual(tanh(1000), 1.0)
        self.assertEqual(tanh(-1000), -1.0)


if __name__ == "__main__":
    unittest.main()
