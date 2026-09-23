import unittest

from ndlm.main import plateau_stop_criterion


class PlateauEarlyStopTests(unittest.TestCase):
    def test_plateau_stop_when_loss_and_grad_are_stagnant(self):
        should_stop, reason = plateau_stop_criterion(
            current_loss=1.0,
            best_loss=0.999,
            grad_norm=1e-7,
            stop_patience=2,
            min_delta=1e-4,
            grad_norm_threshold=1e-5,
            mode="loss_and_grad_plateau",
            plateau_count=2,
            grad_plateau_count=2,
        )
        self.assertTrue(should_stop)
        self.assertTrue("loss" in reason.lower() or "gradient" in reason.lower())

    def test_plateau_stop_is_false_when_loss_is_still_improving(self):
        should_stop, _ = plateau_stop_criterion(
            current_loss=0.8,
            best_loss=0.99,
            grad_norm=1e-3,
            stop_patience=2,
            min_delta=1e-4,
            grad_norm_threshold=1e-5,
            mode="loss_and_grad_plateau",
            plateau_count=1,
            grad_plateau_count=0,
        )
        self.assertFalse(should_stop)


if __name__ == "__main__":
    unittest.main()
