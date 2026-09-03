"""
Human-like Intelligence Module
Implements decision-making, value shaping, social intelligence, and meta-cognition
Based on advanced AI techniques from cognitive science and behavioral economics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecisionState:
    """State for decision-making components."""
    confidence: float = 0.5
    self_efficacy: float = 0.5
    regulatory_resource: float = 1.0
    emotion_valence: Optional[np.ndarray] = None
    trust_level: float = 0.5


class ProspectTheoryUtility:
    """
    Prospect Theory: Loss aversion and probability weighting
    Humans are more sensitive to losses than gains
    """
    
    def __init__(self, loss_aversion: float = 2.25, alpha: float = 0.88, gamma: float = 0.61):
        self.loss_aversion = loss_aversion
        self.alpha = alpha
        self.gamma = gamma
    
    def value_function(self, x: float) -> float:
        """Value function with loss aversion."""
        if x >= 0:
            return x ** self.alpha
        else:
            return -self.loss_aversion * ((-x) ** self.alpha)
    
    def probability_weighting(self, p: float) -> float:
        """Probability weighting function."""
        return (p ** self.gamma) / ((p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma))
    
    def prospect_utility(self, rewards: np.ndarray, probabilities: np.ndarray) -> float:
        """Compute prospect utility from outcomes and probabilities."""
        # Sort by reward (ascending) for cumulative weighting
        sorted_idx = np.argsort(rewards)
        rewards = rewards[sorted_idx]
        probs = probabilities[sorted_idx]
        
        # Cumulative prospect theory
        cum_probs = np.cumsum(probs)
        weights = np.array([
            self.probability_weighting(cum_probs[i]) - self.probability_weighting(cum_probs[i-1]) 
            if i > 0 else self.probability_weighting(probs[0])
            for i in range(len(probs))
        ])
        
        values = np.array([self.value_function(r) for r in rewards])
        return np.sum(weights * values)


class HyperbolicDiscounting:
    """
    Hyperbolic Discounting: Time preference modeling
    Humans devalue future rewards hyperbolically, not exponentially
    """
    
    def __init__(self, k: float = 0.1):
        self.k = k  # Discount rate
    
    def present_value(self, reward: float, delay: int) -> float:
        """Compute present value of delayed reward."""
        return reward / (1 + self.k * delay)
    
    def sequence_value(self, rewards: List[float], delays: List[int]) -> float:
        """Compute value of a sequence of rewards."""
        return sum([self.present_value(r, d) for r, d in zip(rewards, delays)])


class CuriosityModule:
    """
    Curiosity: Intrinsic reward from prediction error
    Encourages exploration of novel states
    """
    
    def __init__(self, eta: float = 0.5):
        self.eta = eta  # Curiosity weight
        self.forward_model = None  # Could be a neural network in full implementation
    
    def curiosity_reward(self, predicted_state: np.ndarray, actual_state: np.ndarray) -> float:
        """Compute curiosity reward from prediction error."""
        # Simple MSE for now, could use neural forward model
        prediction_error = np.mean((predicted_state - actual_state) ** 2)
        return self.eta * prediction_error
    
    def total_reward(self, extrinsic_reward: float, curiosity_bonus: float) -> float:
        """Combine extrinsic and intrinsic rewards."""
        return extrinsic_reward + curiosity_bonus


class AffectHeuristic:
    """
    Affect Heuristic: Fast emotional decisions under time pressure
    Emotions override rational decisions when time is limited
    """
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma  # Emotional weight
    
    def emotional_decision(self, rational_utility: np.ndarray, 
                           emotional_valence: np.ndarray, 
                           time_pressure: float) -> np.ndarray:
        """
        Combine rational and emotional utilities based on time pressure.
        
        Args:
            rational_utility: Rational Q-values
            emotional_valence: Emotional valence for each action
            time_pressure: 0-1 scale (1 = high pressure)
        
        Returns:
            Combined utility
        """
        if time_pressure > 0.8:
            # High stress: use emotion only
            return emotional_valence
        else:
            # Mixed policy
            combined = rational_utility + self.gamma * 0.3 * emotional_valence
            return combined


class TheoryOfMind:
    """
    Theory of Mind: Belief inference about other agents
    Simulates what others might think or want
    """
    
    def __init__(self, num_hypotheses: int = 5):
        self.num_hypotheses = num_hypotheses
        self.belief_state = np.ones(num_hypotheses) / num_hypotheses
    
    def infer_belief(self, observed_action: int, state: np.ndarray, 
                     model_of_other: Optional[nn.Module] = None) -> np.ndarray:
        """
        Update belief about other agent's hidden goal.
        
        Args:
            observed_action: Action taken by other agent
            state: Current state
            model_of_other: Neural network simulating other agent
        
        Returns:
            Updated belief distribution
        """
        if model_of_other is None:
            # Simple heuristic: uniform update
            self.belief_state = np.ones(self.num_hypotheses) / self.num_hypotheses
        else:
            # Likelihood-based update
            for goal_idx in range(self.num_hypotheses):
                # Predict what other would do with this goal
                with torch.no_grad():
                    predicted_q = model_of_other(torch.tensor(state).unsqueeze(0))
                    likelihood = torch.exp(predicted_q[0, observed_action].item())
                self.belief_state[goal_idx] += likelihood
        
        # Normalize
        self.belief_state = self.belief_state / self.belief_state.sum()
        return self.belief_state
    
    def predict_action(self, state: np.ndarray, model_of_other: nn.Module) -> int:
        """Predict other agent's next action based on belief."""
        weighted_q = np.zeros(model_of_other.output_dim)
        
        for goal_idx in range(self.num_hypotheses):
            with torch.no_grad():
                q = model_of_other(torch.tensor(state).unsqueeze(0))
                weighted_q += self.belief_state[goal_idx] * q[0].cpu().numpy()
        
        return np.argmax(weighted_q)


class TrustModel:
    """
    Trust Model: Beta distribution over cooperation probability
    Tracks reliability of other agents over time
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, forgetting_rate: float = 0.99):
        self.alpha = alpha
        self.beta = beta
        self.forgetting_rate = forgetting_rate
    
    def update(self, cooperated: bool):
        """Update trust based on cooperation observation."""
        if cooperated:
            self.alpha += 1
        else:
            self.beta += 1
    
    def expected_trust(self) -> float:
        """Get expected trust level."""
        return self.alpha / (self.alpha + self.beta)
    
    def uncertainty(self) -> float:
        """Get uncertainty in trust estimate."""
        return (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
    
    def time_decay(self):
        """Apply time decay to trust (forgetting)."""
        self.alpha *= self.forgetting_rate
        self.beta *= self.forgetting_rate


class Metacognition:
    """
    Metacognition: Confidence calibration and deliberation decision
    Monitors own performance and decides when to think harder
    """
    
    def __init__(self, initial_confidence: float = 0.5, alpha: float = 0.1):
        self.confidence = initial_confidence
        self.alpha = alpha
    
    def update_confidence(self, was_correct: bool):
        """Update confidence based on outcome."""
        if was_correct:
            self.confidence += self.alpha * (1.0 - self.confidence)
        else:
            self.confidence += self.alpha * (0.0 - self.confidence)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
    
    def should_deliberate(self, task_difficulty: float = 0.5) -> bool:
        """
        Decide whether to deliberate more on this task.
        
        Args:
            task_difficulty: 0-1 scale
        
        Returns:
            True if should deliberate more
        """
        return (self.confidence < 0.3) or (task_difficulty > 0.7)


class SelfEfficacy:
    """
    Self-Efficacy: Belief in own ability
    Asymmetric learning from success vs failure
    """
    
    def __init__(self, initial: float = 0.5, alpha: float = 0.2, success_boost: float = 1.5):
        self.efficacy = initial
        self.alpha = alpha
        self.success_boost = success_boost
    
    def update(self, outcome_success: bool):
        """Update self-efficacy based on outcome."""
        adjusted_alpha = self.alpha * (1 + self.success_boost if outcome_success else 1.0)
        self.efficacy += adjusted_alpha * (1.0 if outcome_success else 0.0 - self.efficacy)
        self.efficacy = np.clip(self.efficacy, 0.0, 1.0)
    
    def modulate_risk_taking(self, base_risk: float) -> float:
        """Modulate risk-taking based on self-efficacy."""
        return base_risk * (1 + self.efficacy)


class CooperationIndex:
    """
    Cooperation Index: Probability of cooperating in social dilemmas
    Based on future value and trust
    """
    
    def __init__(self, beta: float = 2.0, theta: float = 0.3):
        self.beta = beta
        self.theta = theta
    
    def cooperation_probability(self, benefit_b: float, cost_c: float, 
                               trust: float, discount_gamma: float) -> float:
        """
        Compute probability of cooperating.
        
        Args:
            benefit_b: Future benefit from cooperation
            cost_c: Current cost of cooperation
            trust: Trust level in partner
            discount_gamma: Discount factor for future value
        
        Returns:
            Cooperation probability (0-1)
        """
        # Future value of relationship
        future_value = (discount_gamma * benefit_b) / (1 - discount_gamma)
        net_utility = future_value - cost_c + trust * 0.5
        
        # Sigmoid function
        return 1 / (1 + np.exp(-self.beta * (net_utility - self.theta)))


class SocialUtility:
    """
    Social Utility: Combines self-interest, norms, others' welfare, and risk
    Prevents sociopathic behavior
    """
    
    def __init__(self, beta_self: float = 1.0, beta_norm: float = 0.5, 
                 beta_other: float = 0.3, beta_risk: float = 0.4):
        self.beta_self = beta_self
        self.beta_norm = beta_norm
        self.beta_other = beta_other
        self.beta_risk = beta_risk
    
    def compute_social_utility(self, action: int, state: np.ndarray, 
                               q_net: nn.Module, norm_score: np.ndarray,
                               empathy_sim: np.ndarray, risk: float) -> float:
        """
        Compute social utility for an action.
        
        Args:
            action: Action index
            state: Current state
            q_net: Q-network for self-interest
            norm_score: Social norm scores for actions
            empathy_sim: Empathy similarity scores for actions
            risk: Risk associated with action
        
        Returns:
            Social utility
        """
        with torch.no_grad():
            U_self = q_net(torch.tensor(state).unsqueeze(0))[0, action].item()
        
        U_norm = norm_score[action]
        U_other = empathy_sim[action]
        
        return (self.beta_self * U_self + 
                self.beta_norm * U_norm + 
                self.beta_other * U_other - 
                self.beta_risk * risk)


class EmpathyModule:
    """
    Empathy: Perspective taking and emotion simulation
    Blends self and other emotions for prosocial behavior
    """
    
    def __init__(self, emotion_dim: int = 4):
        self.emotion_dim = emotion_dim
        self.W_pt = np.random.randn(emotion_dim, 128)  # Perspective mapping
    
    def simulate_other_emotion(self, other_state: np.ndarray, 
                               event_embedding: np.ndarray) -> np.ndarray:
        """
        Simulate what the other person feels.
        
        Args:
            other_state: Other agent's state
            event_embedding: Embedding of the event
        
        Returns:
            Simulated emotion vector
        """
        combined = np.concatenate([other_state, event_embedding])
        raw_emo = self.W_pt @ combined
        return self.softmax(raw_emo)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def blend_emotions(self, self_emo: np.ndarray, other_emo: np.ndarray, 
                      similarity_score: float) -> np.ndarray:
        """
        Blend self and other emotions based on similarity.
        
        Args:
            self_emo: Self's emotion
            other_emo: Simulated other's emotion
            similarity_score: How similar the other is (0-1)
        
        Returns:
            Blended emotion
        """
        return similarity_score * other_emo + (1 - similarity_score) * self_emo


class EmotionRegulator:
    """
    Emotion Regulation: Reappraisal to alter emotional responses
    Reframes events to reduce negative emotional impact
    """
    
    def __init__(self, emotion_dim: int = 4):
        self.emotion_dim = emotion_dim
        self.W_r = np.random.randn(emotion_dim, emotion_dim + 64)
    
    def reappraise(self, raw_appraisal: np.ndarray, context_embedding: np.ndarray) -> np.ndarray:
        """
        Reappraise emotional response based on context.
        
        Args:
            raw_appraisal: Raw emotional appraisal [relevance, congruence, coping, novelty]
            context_embedding: Context information
        
        Returns:
            Regulated emotional appraisal
        """
        combined = np.concatenate([raw_appraisal, context_embedding])
        new_appraisal = np.maximum(0, self.W_r @ combined)  # ReLU
        return self.softmax(new_appraisal)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class HumanLikeIntelligenceSuite:
    """
    Complete suite of human-like intelligence components
    Integrates all decision-making, social, and meta-cognitive modules
    """
    
    def __init__(self):
        logger.info("Initializing Human-like Intelligence Suite")
        
        # Decision Engine
        self.prospect_theory = ProspectTheoryUtility()
        self.hyperbolic_discounting = HyperbolicDiscounting()
        self.curiosity = CuriosityModule()
        self.affect_heuristic = AffectHeuristic()
        
        # Social Intelligence
        self.theory_of_mind = TheoryOfMind()
        self.trust_model = TrustModel()
        self.cooperation_index = CooperationIndex()
        self.social_utility = SocialUtility()
        self.empathy = EmpathyModule()
        self.emotion_regulator = EmotionRegulator()
        
        # Meta-Cognition
        self.metacognition = Metacognition()
        self.self_efficacy = SelfEfficacy()
        
        # Decision State
        self.decision_state = DecisionState()
        
        logger.info("Human-like Intelligence Suite initialized")
    
    def get_decision_state(self) -> DecisionState:
        """Get current decision state."""
        return self.decision_state
    
    def update_state(self, outcome_success: bool, task_difficulty: float = 0.5):
        """Update internal state based on outcome."""
        self.metacognition.update_confidence(outcome_success)
        self.self_efficacy.update(outcome_success)
        
        # Trust decay
        self.trust_model.time_decay()
    
    def compute_enhanced_utility(self, raw_utility: float, 
                                 time_pressure: float = 0.0,
                                 social_context: Optional[Dict] = None) -> float:
        """
        Compute enhanced utility with all human-like components.
        
        Args:
            raw_utility: Basic Q-value or utility
            time_pressure: 0-1 scale (high = stressful)
            social_context: Optional social context information
        
        Returns:
            Enhanced utility value
        """
        # Apply prospect theory transformation
        if isinstance(raw_utility, (list, np.ndarray)):
            enhanced = self.prospect_theory.prospect_utility(
                np.array(raw_utility), 
                np.ones(len(raw_utility)) / len(raw_utility)
            )
        else:
            enhanced = self.prospect_theory.value_function(raw_utility)
        
        # Apply hyperbolic discounting for future rewards
        if social_context and 'delay' in social_context:
            enhanced = self.hyperbolic_discounting.present_value(enhanced, social_context['delay'])
        
        # Apply affect heuristic under time pressure
        if self.decision_state.emotion_valence is not None:
            enhanced = self.affect_heuristic.emotional_decision(
                np.array([enhanced]),
                self.decision_state.emotion_valence,
                time_pressure
            )[0]
        
        return enhanced
    
    def should_cooperate(self, benefit: float, cost: float, 
                         partner_trust: float = 0.5) -> bool:
        """
        Decide whether to cooperate in social situation.
        
        Args:
            benefit: Future benefit from cooperation
            cost: Current cost of cooperation
            partner_trust: Trust level in partner
        
        Returns:
            True if should cooperate
        """
        coop_prob = self.cooperation_index.cooperation_probability(
            benefit, cost, partner_trust, discount_gamma=0.9
        )
        return np.random.random() < coop_prob
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics for monitoring."""
        return {
            "prospect_theory": {"loss_aversion": self.prospect_theory.loss_aversion},
            "metacognition": {"confidence": self.metacognition.confidence},
            "self_efficacy": {"efficacy": self.self_efficacy.efficacy},
            "trust": {"expected_trust": self.trust_model.expected_trust()},
            "decision_state": {
                "confidence": self.decision_state.confidence,
                "self_efficacy": self.decision_state.self_efficacy,
                "regulatory_resource": self.decision_state.regulatory_resource
            }
        }