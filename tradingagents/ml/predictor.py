"""ML predictor for discovery pipeline — loads trained model and runs inference.

Gracefully degrades: if no model file exists, all predictions return None.
The discovery pipeline works exactly as before without a trained model.

Classes:
    LGBMWrapper          — sklearn-compatible 3-class LightGBM wrapper
    MLPredictor          — single-model inference; auto-detects ensemble config
    EnsemblePredictor    — averages win_prob across multiple MLPredictors
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Default model path relative to project root
DEFAULT_MODEL_DIR = Path("data/ml")
MODEL_FILENAME = "tabpfn_model.pkl"
METRICS_FILENAME = "metrics.json"
ENSEMBLE_CONFIG_FILENAME = "ensemble_config.json"

# Class label mapping
LABEL_MAP = {-1: "LOSS", 0: "TIMEOUT", 1: "WIN"}


class LGBMWrapper:
    """Sklearn-compatible wrapper for LightGBM booster with original label mapping.

    Defined here (not in train script) so pickle can find the class on deserialization.
    """

    def __init__(self, booster, y_train=None):
        self.booster = booster
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.booster.predict(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        mapped = np.argmax(probas, axis=1)
        return self.classes_[mapped]


class MLPredictor:
    """Wraps a trained ML model for win probability prediction.

    Usage:
        predictor = MLPredictor.load()  # loads from default path
        if predictor is not None:
            result = predictor.predict(feature_dict)
            # result = {"win_prob": 0.73, "loss_prob": 0.12, "timeout_prob": 0.15, "prediction": "WIN"}
    """

    def __init__(self, model: Any, feature_columns: List[str], model_type: str = "tabpfn"):
        self.model = model
        self.feature_columns = feature_columns
        self.model_type = model_type

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> Optional[MLPredictor]:
        """Load a trained model from disk. Returns None if no model exists.

        Checks for ensemble_config.json first — if found, returns an EnsemblePredictor
        (which is a drop-in replacement). Falls back to a single-model MLPredictor.
        """
        if model_dir is None:
            model_dir = str(DEFAULT_MODEL_DIR)

        # Check for ensemble config first
        ensemble_config_path = os.path.join(model_dir, ENSEMBLE_CONFIG_FILENAME)
        if os.path.exists(ensemble_config_path):
            try:
                ep = EnsemblePredictor.load_from_config(ensemble_config_path)
                if ep is not None:
                    logger.info(f"Loaded EnsemblePredictor from {ensemble_config_path}")
                    return ep  # type: ignore[return-value]
            except Exception as e:
                logger.warning(f"Failed to load ensemble config: {e} — falling back to single model")

        model_path = os.path.join(model_dir, MODEL_FILENAME)
        if not os.path.exists(model_path):
            logger.debug(f"No ML model found at {model_path} — ML predictions disabled")
            return None

        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)

            model = saved["model"]
            feature_columns = saved.get("feature_columns", FEATURE_COLUMNS)
            model_type = saved.get("model_type", "unknown")

            logger.info(f"Loaded ML model ({model_type}) from {model_path}")
            return cls(model=model, feature_columns=feature_columns, model_type=model_type)

        except Exception as e:
            logger.warning(f"Failed to load ML model from {model_path}: {e}")
            return None

    def predict(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Predict win probability for a single candidate.

        Args:
            features: Dict mapping feature names to values (from feature_engineering).

        Returns:
            Dict with win_prob, loss_prob, timeout_prob, prediction, or None on error.
        """
        try:
            # Build feature vector in correct order
            X = np.array([[features.get(col, np.nan) for col in self.feature_columns]])
            X_df = pd.DataFrame(X, columns=self.feature_columns)

            # Get probability predictions
            probas = self.model.predict_proba(X_df)

            # Map class indices to labels
            # Model classes should be [-1, 0, 1] or [0, 1, 2] depending on training
            classes = list(self.model.classes_)

            # Build probability dict
            result: Dict[str, Any] = {}
            for i, cls_label in enumerate(classes):
                prob = float(probas[0][i])
                if cls_label == 1 or cls_label == 2:  # WIN class
                    result["win_prob"] = prob
                elif cls_label == -1 or cls_label == 0:
                    if cls_label == -1:
                        result["loss_prob"] = prob
                    else:
                        # Could be timeout (0) in {-1,0,1} or loss in {0,1,2}
                        if len(classes) == 3 and max(classes) == 2:
                            result["loss_prob"] = prob
                        else:
                            result["timeout_prob"] = prob

            # Ensure all keys present
            result.setdefault("win_prob", 0.0)
            result.setdefault("loss_prob", 0.0)
            result.setdefault("timeout_prob", 0.0)

            # Predicted class
            pred_idx = np.argmax(probas[0])
            pred_class = classes[pred_idx]
            result["prediction"] = LABEL_MAP.get(pred_class, str(pred_class))

            return result

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def predict_batch(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Predict win probabilities for multiple candidates."""
        return [self.predict(f) for f in feature_dicts]

    def save(self, model_dir: Optional[str] = None, suffix: str = "") -> str:
        """Save the model to disk.
        
        Args:
            model_dir: Directory to save the model in
            suffix: Optional suffix to append to filename (e.g., "_tabpfn" for "tabpfn_model.pkl")
        """
        if model_dir is None:
            model_dir = str(DEFAULT_MODEL_DIR)

        os.makedirs(model_dir, exist_ok=True)
        
        # Build filename with optional suffix
        if suffix:
            model_filename = MODEL_FILENAME.replace(".pkl", f"{suffix}.pkl")
        else:
            model_filename = MODEL_FILENAME
        model_path = os.path.join(model_dir, model_filename)

        saved = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
        }

        with open(model_path, "wb") as f:
            pickle.dump(saved, f)

        logger.info(f"Saved ML model to {model_path}")
        return model_path


class EnsemblePredictor:
    """Ensemble of multiple MLPredictors — averages win_prob across all base models.

    Acts as a drop-in replacement for MLPredictor anywhere predict()/predict_batch()
    are called. Optionally uses a stacking meta-model fitted by ensemble_models.py.

    Usage:
        ep = EnsemblePredictor.load_from_config("data/ml/ensemble_config.json")
        if ep is not None:
            result = ep.predict(feature_dict)
    """

    def __init__(
        self,
        predictors: List[MLPredictor],
        weights: Optional[List[float]] = None,
        stacking_model: Optional[Any] = None,
        model_type: str = "ensemble",
    ):
        self.predictors = predictors
        self.weights = np.array(weights) if weights else np.ones(len(predictors))
        self.weights = self.weights / self.weights.sum()
        self.stacking_model = stacking_model
        self.model_type = model_type
        # Expose classes_ so evaluate() in train_ml_model.py can use it
        self.classes_ = np.array([-1, 0, 1])

    @classmethod
    def load_from_config(cls, config_path: str) -> Optional["EnsemblePredictor"]:
        """Load an EnsemblePredictor from a JSON config file.

        The config is produced by scripts/ensemble_models.py and has the form:
            {
                "model_paths": ["data/ml/m1.pkl", ...],
                "weights": [0.5, 0.5],
                "has_stacking": false,
                "stacking_model_path": null
            }

        Returns None if any model file is missing.
        """
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Cannot load ensemble config {config_path}: {e}")
            return None

        model_paths = config.get("model_paths", [])
        weights = config.get("weights", None)
        has_stacking = config.get("has_stacking", False)
        stacking_path = config.get("stacking_model_path", None)

        predictors: List[MLPredictor] = []
        for path in model_paths:
            if not os.path.exists(path):
                logger.warning(f"Ensemble model not found: {path}")
                return None
            try:
                with open(path, "rb") as f:
                    saved = pickle.load(f)
                predictor = MLPredictor(
                    model=saved["model"],
                    feature_columns=saved.get("feature_columns", FEATURE_COLUMNS),
                    model_type=saved.get("model_type", "unknown"),
                )
                predictors.append(predictor)
                logger.debug(f"Loaded ensemble member: {predictor.model_type} from {path}")
            except Exception as e:
                logger.warning(f"Failed to load ensemble member {path}: {e}")
                return None

        stacking_model = None
        if has_stacking and stacking_path and os.path.exists(stacking_path):
            try:
                with open(stacking_path, "rb") as f:
                    stacking_data = pickle.load(f)
                stacking_model = stacking_data["model"]
                logger.debug(f"Loaded stacking meta-model from {stacking_path}")
            except Exception as e:
                logger.warning(f"Failed to load stacking model: {e}")

        return cls(predictors=predictors, weights=weights, stacking_model=stacking_model)

    def _get_win_probs_from_predictor(self, predictor: MLPredictor, X_df: pd.DataFrame) -> float:
        """Extract scalar P(WIN) from a single predictor for one sample."""
        probas = predictor.model.predict_proba(X_df)  # shape (1, n_classes)
        classes = list(predictor.model.classes_)
        if 1 in classes:
            return float(probas[0][classes.index(1)])
        # Binary model [0, 1]: WIN is index 1
        return float(probas[0][1])

    def predict(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Predict win probability by averaging across all base predictors.

        Args:
            features: Dict mapping feature names to values.

        Returns:
            Dict with win_prob (ensemble average), loss_prob, timeout_prob, prediction.
        """
        try:
            individual_win_probs = []
            for predictor in self.predictors:
                X = np.array([[features.get(col, np.nan) for col in predictor.feature_columns]])
                X_df = pd.DataFrame(X, columns=predictor.feature_columns)
                wp = self._get_win_probs_from_predictor(predictor, X_df)
                individual_win_probs.append(wp)

            # Weighted average
            win_prob = float(np.dot(self.weights, individual_win_probs))

            # Use stacking meta-model if available
            if self.stacking_model is not None:
                prob_vec = np.array(individual_win_probs).reshape(1, -1)
                stacked = self.stacking_model.predict_proba(prob_vec)
                classes = list(self.stacking_model.classes_)
                win_idx = classes.index(1) if 1 in classes else 1
                win_prob = float(stacked[0][win_idx])

            prediction = "WIN" if win_prob >= 0.5 else "NOT_WIN"
            return {
                "win_prob": win_prob,
                "loss_prob": 1.0 - win_prob,  # approximate in binary ensemble
                "timeout_prob": 0.0,
                "prediction": prediction,
            }
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
            return None

    def predict_batch(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Predict win probabilities for multiple candidates."""
        return [self.predict(f) for f in feature_dicts]
