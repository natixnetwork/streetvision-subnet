from typing import Any, Dict, List, Tuple
import hashlib
import numpy as np

import bittensor as bt
from PIL import Image
from sentence_transformers import SentenceTransformer

from natix.validator.organic_validator import OrganicValidator


class OrganicEvaluator:
    """
    Evaluates miner responses to organic requests using Florence-2 validation
    and anti-collusion mechanisms.
    """

    def __init__(
        self,
        organic_validator: OrganicValidator,
        performance_trackers: Dict[str, Any],
        sample_rate: float = 0.15,
        similarity_threshold: float = 0.9,
        explanation_required_threshold: float = 0.5,
        weights: Dict[str, float] = None,
        penalties: Dict[str, float] = None
    ):
        self.organic_validator = organic_validator
        self.performance_trackers = performance_trackers
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.explanation_required_threshold = explanation_required_threshold

        self.weights = weights or {
            'prediction_accuracy': 0.4,
            'object_verification': 0.3,
            'explanation_quality': 0.2,
            'diversity_bonus': 0.1
        }

        self.penalties = penalties or {
            'missing_explanation': 0.0,
            'false_claim': -0.3,
            'high_similarity': -0.2
        }

        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def should_validate(self, task_hash: str) -> bool:
        """
        Determine if this request should get deep validation based on sampling rate.
        """
        hash_int = int(task_hash[:8], 16)
        return (hash_int % 100) < (self.sample_rate * 100)

    def check_explanation_diversity(self, explanations: List[str]) -> List[float]:
        """
        Calculate diversity scores for explanations using semantic similarity.
        Returns list of diversity scores (0-1, higher is more diverse).
        """
        if len(explanations) <= 1:
            return [1.0] * len(explanations)

        valid_explanations = [exp for exp in explanations if exp and exp.strip()]

        if len(valid_explanations) <= 1:
            return [1.0] * len(explanations)

        embeddings = self.sentence_model.encode(valid_explanations)

        diversity_scores = []
        for i, exp in enumerate(explanations):
            if not exp or not exp.strip():
                diversity_scores.append(1.0)
                continue

            max_similarity = 0.0
            for j, other_exp in enumerate(valid_explanations):
                if i != j:
                    cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    max_similarity = max(max_similarity, cos_sim)

            diversity = 1.0 - max_similarity
            diversity_scores.append(diversity)

        return diversity_scores

    def calculate_reward(
        self,
        miner_prediction: float,
        miner_explanation: str,
        verification_result: Dict,
        diversity_score: float
    ) -> Tuple[float, Dict]:
        """
        Calculate reward for a miner's organic request response.

        Returns:
            Tuple of (reward_score, metrics_dict)
        """
        florence_pred = verification_result['florence_prediction']
        objects_verified = verification_result['objects_verified']
        specificity_score = verification_result['specificity_score']
        false_claims = verification_result['false_claims']

        if miner_prediction > self.explanation_required_threshold and not miner_explanation.strip():
            return self.penalties['missing_explanation'], {
                'reason': 'missing_explanation',
                'prediction_accuracy': 0.0,
                'object_verification': 0.0,
                'explanation_quality': 0.0,
                'diversity_bonus': 0.0
            }

        prediction_diff = abs(miner_prediction - florence_pred)
        prediction_accuracy = max(0.0, 1.0 - prediction_diff)

        object_verification_score = objects_verified

        explanation_quality = specificity_score if miner_explanation.strip() else 0.0

        diversity_bonus = diversity_score if diversity_score < (1.0 - self.similarity_threshold) else 0.0

        reward = (
            self.weights['prediction_accuracy'] * prediction_accuracy +
            self.weights['object_verification'] * object_verification_score +
            self.weights['explanation_quality'] * explanation_quality +
            self.weights['diversity_bonus'] * diversity_bonus
        )

        if false_claims:
            reward += self.penalties['false_claim'] * len(false_claims)

        if diversity_score < (1.0 - self.similarity_threshold):
            reward += self.penalties['high_similarity']

        reward = max(0.0, min(1.0, reward))

        metrics = {
            'prediction_accuracy': prediction_accuracy,
            'object_verification': object_verification_score,
            'explanation_quality': explanation_quality,
            'diversity_bonus': diversity_bonus,
            'false_claims_count': len(false_claims),
            'diversity_score': diversity_score
        }

        return reward, metrics

    def evaluate_responses(
        self,
        image: Image.Image,
        responses: List[Dict],
        uids: List[int],
        axons: List
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Evaluate miner responses with full Florence-2 validation.

        Args:
            image: PIL Image
            responses: List of dicts with 'result' (synapse) and 'miner_uid'
            uids: List of miner UIDs
            axons: List of miner axons

        Returns:
            Tuple of (rewards array, metrics list)
        """
        predictions = []
        explanations = []

        for response in responses:
            synapse = response.get('result')
            if synapse:
                predictions.append(getattr(synapse, 'prediction', -1.0))
                explanations.append(getattr(synapse, 'explanation', ""))
            else:
                predictions.append(-1.0)
                explanations.append("")

        diversity_scores = self.check_explanation_diversity(explanations)

        rewards = []
        all_metrics = []

        for i, (uid, pred, explanation, diversity) in enumerate(zip(uids, predictions, explanations, diversity_scores)):
            try:
                if pred == -1.0:
                    rewards.append(0.0)
                    all_metrics.append({})
                    continue

                verification_result = self.organic_validator.verify_explanation(
                    image, explanation, pred
                )

                reward, metrics = self.calculate_reward(
                    pred, explanation, verification_result, diversity
                )

                rewards.append(reward)
                all_metrics.append(metrics)

                axon = axons[i]
                miner_hotkey = axon.hotkey
                tracker = self.performance_trackers["image"]

                tracked_hotkeys = tracker.miner_hotkeys
                if uid in tracked_hotkeys and tracked_hotkeys[uid] != miner_hotkey:
                    bt.logging.info(f"Miner hotkey changed for UID {uid}. Resetting performance metrics.")
                    tracker.reset_miner_history(uid, miner_hotkey)

                tracker.update(uid, pred, verification_result['florence_prediction'], miner_hotkey)

                bt.logging.info(
                    f"[ORGANIC] UID {uid} | Pred: {pred:.2f} | Florence: {verification_result['florence_prediction']:.2f} | Reward: {reward:.2f}"
                )

            except Exception as e:
                bt.logging.error(f"Error evaluating miner {uid}: {e}")
                rewards.append(0.0)
                all_metrics.append({})

        return np.array(rewards), all_metrics

    def evaluate_lightweight(
        self,
        responses: List[Dict],
        uids: List[int]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Lightweight evaluation for non-sampled requests.
        Only checks validity, no Florence-2 inference.
        """
        rewards = []
        metrics = []

        for response in responses:
            synapse = response.get('result')
            if not synapse:
                rewards.append(0.0)
                metrics.append({})
                continue

            pred = getattr(synapse, 'prediction', -1.0)
            explanation = getattr(synapse, 'explanation', "")

            if pred == -1.0:
                rewards.append(0.0)
                metrics.append({'reason': 'invalid_prediction'})
            elif pred > self.explanation_required_threshold and not explanation.strip():
                rewards.append(self.penalties['missing_explanation'])
                metrics.append({'reason': 'missing_explanation'})
            else:
                rewards.append(0.5)
                metrics.append({'reason': 'lightweight_validation'})

        return np.array(rewards), metrics
