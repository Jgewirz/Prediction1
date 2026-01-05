"""
Calibration tracking system for measuring probability extraction accuracy over time.

This module tracks predicted probabilities vs actual outcomes to:
- Calculate Brier scores (calibration accuracy)
- Measure directional accuracy (win rate)
- Identify systematic biases
- Enable confidence-weighted improvements
"""
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger


class CalibrationTracker:
    """Track probability extraction accuracy for continuous improvement."""

    def __init__(self, storage_path: str = "calibration_data.json"):
        """
        Initialize calibration tracker.

        Args:
            storage_path: Path to JSON file for persistent storage
        """
        self.storage_path = Path(storage_path)
        self.data: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load existing calibration data from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded {len(self.data)} calibration records from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Could not load calibration data: {e}")
                self.data = []
        else:
            self.data = []

    def _save(self):
        """Save calibration data to disk."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save calibration data: {e}")

    def record_prediction(self, ticker: str, predicted_prob: float, market_price: float,
                         confidence: float, r_score: float, action: str,
                         event_ticker: Optional[str] = None,
                         reasoning: Optional[str] = None) -> str:
        """
        Record a prediction before outcome is known.

        Args:
            ticker: Market ticker
            predicted_prob: Research probability (0-100 scale)
            market_price: Market implied probability (0-100 scale)
            confidence: Model confidence in prediction (0-1)
            r_score: Risk-adjusted edge score
            action: Betting action (buy_yes, buy_no, skip)
            event_ticker: Parent event ticker
            reasoning: Model's reasoning for the prediction

        Returns:
            Prediction ID for later outcome recording
        """
        prediction_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = {
            "prediction_id": prediction_id,
            "ticker": ticker,
            "event_ticker": event_ticker,
            "predicted_prob": predicted_prob,
            "market_price": market_price,
            "confidence": confidence,
            "r_score": r_score,
            "action": action,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "outcome": None,  # To be filled when market resolves
            "resolved_timestamp": None,
            "actual_payout": None
        }

        self.data.append(record)
        self._save()
        logger.debug(f"Recorded prediction for {ticker}: prob={predicted_prob:.1f}%, conf={confidence:.2f}")

        return prediction_id

    def record_outcome(self, ticker: str, outcome: bool,
                       payout: Optional[float] = None) -> bool:
        """
        Record actual outcome when market resolves.

        Args:
            ticker: Market ticker
            outcome: True if YES won, False if NO won
            payout: Actual payout amount (optional)

        Returns:
            True if outcome was recorded, False if prediction not found
        """
        # Find most recent unresolved prediction for this ticker
        for entry in reversed(self.data):
            if entry["ticker"] == ticker and entry["outcome"] is None:
                entry["outcome"] = 1.0 if outcome else 0.0
                entry["resolved_timestamp"] = datetime.now().isoformat()
                entry["actual_payout"] = payout
                self._save()
                logger.info(f"Recorded outcome for {ticker}: {'YES' if outcome else 'NO'}")
                return True

        logger.warning(f"No unresolved prediction found for {ticker}")
        return False

    def get_unresolved_predictions(self) -> List[Dict[str, Any]]:
        """Get list of predictions awaiting outcomes."""
        return [e for e in self.data if e["outcome"] is None]

    def get_resolved_predictions(self) -> List[Dict[str, Any]]:
        """Get list of predictions with known outcomes."""
        return [e for e in self.data if e["outcome"] is not None]

    def calculate_brier_score(self, min_confidence: Optional[float] = None) -> Optional[float]:
        """
        Calculate Brier score (lower is better, 0 is perfect).

        Brier score = mean((predicted_prob - outcome)^2)

        Args:
            min_confidence: Only include predictions with confidence >= this value

        Returns:
            Brier score or None if no resolved predictions
        """
        resolved = self.get_resolved_predictions()

        if min_confidence is not None:
            resolved = [e for e in resolved if e["confidence"] >= min_confidence]

        if not resolved:
            return None

        brier = sum(
            ((e["predicted_prob"] / 100) - e["outcome"]) ** 2
            for e in resolved
        ) / len(resolved)

        return brier

    def calculate_accuracy(self, min_confidence: Optional[float] = None) -> Optional[float]:
        """
        Calculate directional accuracy (win rate).

        Args:
            min_confidence: Only include predictions with confidence >= this value

        Returns:
            Accuracy (0-1) or None if no resolved predictions
        """
        resolved = self.get_resolved_predictions()

        if min_confidence is not None:
            resolved = [e for e in resolved if e["confidence"] >= min_confidence]

        if not resolved:
            return None

        correct = sum(
            1 for e in resolved
            if (e["predicted_prob"] > 50) == (e["outcome"] == 1.0)
        )

        return correct / len(resolved)

    def calculate_calibration_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive calibration statistics.

        Returns:
            Dictionary with calibration metrics
        """
        resolved = self.get_resolved_predictions()

        if not resolved:
            return {
                "total_predictions": len(self.data),
                "resolved_predictions": 0,
                "unresolved_predictions": len(self.get_unresolved_predictions()),
                "brier_score": None,
                "accuracy": None,
                "avg_confidence": None,
                "avg_r_score": None,
                "confidence_calibration": None
            }

        # Basic metrics
        brier = self.calculate_brier_score()
        accuracy = self.calculate_accuracy()

        # Average confidence and R-score
        avg_confidence = statistics.mean(e["confidence"] for e in resolved)
        avg_r_score = statistics.mean(e["r_score"] for e in resolved if e["r_score"] is not None)

        # Confidence calibration: compare avg confidence to actual accuracy
        # Well-calibrated means confidence ≈ accuracy
        confidence_calibration = abs(avg_confidence - accuracy) if accuracy else None

        # Bucket analysis: accuracy by confidence range
        buckets = {
            "low_conf (0-0.4)": [e for e in resolved if e["confidence"] < 0.4],
            "med_conf (0.4-0.7)": [e for e in resolved if 0.4 <= e["confidence"] < 0.7],
            "high_conf (0.7-1.0)": [e for e in resolved if e["confidence"] >= 0.7]
        }

        bucket_accuracy = {}
        for bucket_name, bucket_data in buckets.items():
            if bucket_data:
                correct = sum(1 for e in bucket_data if (e["predicted_prob"] > 50) == (e["outcome"] == 1.0))
                bucket_accuracy[bucket_name] = {
                    "count": len(bucket_data),
                    "accuracy": correct / len(bucket_data)
                }

        return {
            "total_predictions": len(self.data),
            "resolved_predictions": len(resolved),
            "unresolved_predictions": len(self.get_unresolved_predictions()),
            "brier_score": brier,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_r_score": avg_r_score,
            "confidence_calibration": confidence_calibration,
            "bucket_accuracy": bucket_accuracy
        }

    def get_bias_analysis(self) -> Dict[str, Any]:
        """
        Analyze systematic biases in predictions.

        Returns:
            Dictionary with bias metrics
        """
        resolved = self.get_resolved_predictions()

        if len(resolved) < 10:
            return {"message": "Need at least 10 resolved predictions for bias analysis"}

        # Calculate average prediction vs average outcome
        avg_predicted = statistics.mean(e["predicted_prob"] / 100 for e in resolved)
        avg_outcome = statistics.mean(e["outcome"] for e in resolved)

        # Overconfidence: predicted higher than outcomes
        overconfidence = avg_predicted - avg_outcome

        # Calculate bias by action type
        buy_yes = [e for e in resolved if e["action"] == "buy_yes"]
        buy_no = [e for e in resolved if e["action"] == "buy_no"]

        return {
            "avg_predicted_prob": avg_predicted,
            "avg_actual_outcome": avg_outcome,
            "overconfidence_bias": overconfidence,
            "buy_yes_accuracy": (sum(1 for e in buy_yes if e["outcome"] == 1.0) / len(buy_yes)) if buy_yes else None,
            "buy_no_accuracy": (sum(1 for e in buy_no if e["outcome"] == 0.0) / len(buy_no)) if buy_no else None,
            "sample_size": len(resolved)
        }

    def print_summary(self):
        """Print a formatted summary of calibration stats."""
        stats = self.calculate_calibration_stats()

        print("\n" + "=" * 50)
        print("CALIBRATION SUMMARY")
        print("=" * 50)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Resolved: {stats['resolved_predictions']}")
        print(f"Unresolved: {stats['unresolved_predictions']}")

        if stats['brier_score'] is not None:
            print(f"\nBrier Score: {stats['brier_score']:.4f} (lower is better)")
            print(f"Accuracy: {stats['accuracy']:.1%}")
            print(f"Avg Confidence: {stats['avg_confidence']:.2f}")
            print(f"Avg R-Score: {stats['avg_r_score']:.2f}")

            if stats['bucket_accuracy']:
                print("\nAccuracy by Confidence Bucket:")
                for bucket, data in stats['bucket_accuracy'].items():
                    print(f"  {bucket}: {data['accuracy']:.1%} (n={data['count']})")

        print("=" * 50 + "\n")


# Global tracker instance for convenience
_tracker: Optional[CalibrationTracker] = None


def get_tracker(storage_path: str = "calibration_data.json") -> CalibrationTracker:
    """Get or create global calibration tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CalibrationTracker(storage_path)
    return _tracker

