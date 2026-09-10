import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parents[1] / "python_code"))

from models import DenseForecaster, gaussian_nll, gaussian_parameters, predict_gaussian, train_forecaster
from plotting import plot_predictions


class ProbabilisticForecastingTests(unittest.TestCase):
    def test_gaussian_nll_produces_positive_finite_sigma(self):
        prediction = torch.zeros(4, 2)
        target = torch.ones(4, 1)

        mean, sigma = gaussian_parameters(prediction)

        self.assertTrue(torch.all(sigma > 0))
        self.assertTrue(torch.isfinite(gaussian_nll(prediction, target)))
        self.assertEqual(mean.shape, sigma.shape)

    def test_training_and_prediction_return_mean_and_sigma(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(24, 6, 1)).astype("float32")
        y = rng.normal(size=(24, 1)).astype("float32")
        model = DenseForecaster(lookback=6, units=8, horizon=1)

        history = train_forecaster(model, X, y, validation_split=0.25, epochs=2, batch_size=8)
        mean, sigma = predict_gaussian(model, X[:3])

        self.assertEqual(mean.shape, (3, 1))
        self.assertEqual(sigma.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(sigma)))
        self.assertTrue(np.all(sigma > 0))
        self.assertEqual(len(history["loss"]), len(history["val_loss"]))

    def test_prediction_plot_accepts_intervals(self):
        actual = np.zeros(4)
        predicted = np.zeros(4)
        lower = -np.ones(4)
        upper = np.ones(4)

        figure, axis = plot_predictions(actual, predicted, lower=lower, upper=upper)

        self.assertEqual(len(axis.collections), 1)
        figure.clear()


if __name__ == "__main__":
    unittest.main()
