import unittest
import numpy as np

from src.fraud_detection.evaluation import (
    select_threshold,
    evaluate_predictions,
    evaluate_at_fpr_targets,
    evaluate_review_capacity,
    format_evaluation_report,
)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # 100 legitimate (low scores), 10 fraud (high scores)
        np.random.seed(42)
        self.y_val = np.array([0] * 100 + [1] * 10)
        self.val_scores = np.concatenate([
            np.random.normal(loc=10.0, scale=2.0, size=100),
            np.random.normal(loc=25.0, scale=2.0, size=10),
        ])

        self.y_test = np.array([0] * 100 + [1] * 10)
        self.test_scores = np.concatenate([
            np.random.normal(loc=10.0, scale=2.0, size=100),
            np.random.normal(loc=25.0, scale=2.0, size=10),
        ])

    def test_select_threshold(self):
        result = select_threshold(self.y_val, self.val_scores)
        self.assertIn("threshold", result)
        self.assertIn("f1", result)
        self.assertGreater(result["f1"], 0.8)

    def test_evaluate_predictions(self):
        metrics = evaluate_predictions(self.y_test, self.test_scores, threshold=18.0)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_evaluate_at_fpr_targets(self):
        fpr_res = evaluate_at_fpr_targets(
            self.y_val, self.val_scores, self.y_test, self.test_scores, targets=[0.05, 0.01]
        )
        self.assertEqual(len(fpr_res), 2)
        self.assertIn("fraud_recall", fpr_res[0])

    def test_evaluate_review_capacity(self):
        review_res = evaluate_review_capacity(self.y_test, self.test_scores, capacities=[10, 20])
        self.assertEqual(len(review_res), 2)
        self.assertEqual(review_res[0]["review_capacity"], 10)
        self.assertGreater(review_res[0]["fraud_found"], 0)

    def test_format_evaluation_report(self):
        metrics = evaluate_predictions(self.y_test, self.test_scores, threshold=18.0)
        report = format_evaluation_report("Test Model", metrics, threshold=18.0)
        self.assertIn("TEST MODEL", report)
        self.assertIn("CLASSIFICATION METRICS", report)


if __name__ == "__main__":
    unittest.main()
