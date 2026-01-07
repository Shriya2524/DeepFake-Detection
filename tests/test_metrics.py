"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest
import torch


class TestMetrics:
    """Tests for metric computation functions."""
    
    def test_compute_classification_metrics(self):
        """Test classification metrics computation."""
        from src.evaluation.metrics import compute_classification_metrics
        
        # Perfect predictions
        predictions = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        targets = torch.tensor([1, 0, 1, 0])
        
        metrics = compute_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        assert metrics['accuracy'] == 1.0
    
    def test_compute_roc_auc(self):
        """Test ROC-AUC computation."""
        from src.evaluation.metrics import compute_roc_auc
        
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        targets = np.array([0, 0, 1, 1])
        
        auc, fpr, tpr, thresholds = compute_roc_auc(probabilities, targets)
        
        assert 0 <= auc <= 1
        assert len(fpr) == len(tpr)
        assert auc == 1.0  # Perfect separation
    
    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        from src.evaluation.metrics import compute_confusion_matrix
        
        predictions = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.3, 0.7], [0.7, 0.3]])
        targets = torch.tensor([1, 0, 1, 0])
        
        cm = compute_confusion_matrix(predictions, targets)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 4
        
        # Check values for perfect predictions
        assert cm[0, 0] == 2  # True negatives
        assert cm[1, 1] == 2  # True positives
        assert cm[0, 1] == 0  # False positives
        assert cm[1, 0] == 0  # False negatives
    
    def test_find_optimal_threshold(self):
        """Test optimal threshold finding."""
        from src.evaluation.metrics import find_optimal_threshold
        
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = np.array([0, 0, 0, 1, 1])
        
        threshold, score = find_optimal_threshold(
            probabilities, targets, metric="f1"
        )
        
        assert 0.1 <= threshold <= 0.9
        assert 0 <= score <= 1
    
    def test_get_predictions(self):
        """Test converting logits/probs to predictions."""
        from src.evaluation.metrics import get_predictions
        
        # Test with 2D logits
        logits = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        preds, probs = get_predictions(logits)
        
        assert preds.tolist() == [1, 1]
        assert len(probs) == 2
    
    def test_evaluate_model(self):
        """Test comprehensive model evaluation."""
        from src.evaluation.metrics import evaluate_model
        
        predictions = torch.tensor([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]
        ])
        targets = torch.tensor([0, 1, 0, 1])
        
        results = evaluate_model(predictions, targets)
        
        assert results.accuracy == 1.0
        assert results.confusion_matrix.shape == (2, 2)
        assert len(results.predictions) == 4


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        from src.evaluation.metrics import EvaluationResults
        
        results = EvaluationResults(
            accuracy=0.95,
            precision=0.93,
            recall=0.92,
            f1=0.925,
            roc_auc=0.97,
            confusion_matrix=np.array([[45, 5], [3, 47]]),
            classification_report="test report",
            predictions=np.array([0, 1, 1, 0]),
            probabilities=np.array([0.1, 0.9, 0.8, 0.2]),
            labels=np.array([0, 1, 1, 0])
        )
        
        d = results.to_dict()
        
        assert d['accuracy'] == 0.95
        assert d['f1'] == 0.925
        assert isinstance(d['confusion_matrix'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

