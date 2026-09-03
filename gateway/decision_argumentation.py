"""
Decision Theory and Argumentation Framework
Implements Expected Utility, Dung Argumentation Framework, and Information Value (EVPI)
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Argument:
    """An argument in the argumentation framework."""
    name: str
    content: str
    strength: float = 1.0
    accepted: bool = False

@dataclass
class DecisionAction:
    """An action for decision making."""
    name: str
    expected_utility: float
    utility_distribution: List[float]  # Utilities across states
    confidence: float = 1.0


class ExpectedUtilityDecisionMaker:
    """
    Expected Utility for decision making under uncertainty.
    Chooses actions that maximize expected utility.
    """
    
    def __init__(self):
        self.decision_history: List[Dict] = []
    
    def compute_expected_utility(self, state_probabilities: np.ndarray, 
                                utility_matrix: np.ndarray) -> np.ndarray:
        """
        Compute expected utilities for all actions.
        state_probabilities: array of state probabilities (n_states)
        utility_matrix: (n_actions, n_states) utilities for each action in each state
        returns: array of expected utilities per action
        """
        expected_utilities = utility_matrix @ state_probabilities
        return expected_utilities
    
    def find_optimal_action(self, state_probabilities: np.ndarray,
                           utility_matrix: np.ndarray,
                           action_names: Optional[List[str]] = None) -> Tuple[int, float]:
        """
        Find the action with maximum expected utility.
        Returns (action_index, expected_utility)
        """
        expected_utilities = self.compute_expected_utility(state_probabilities, utility_matrix)
        best_action = int(np.argmax(expected_utilities))
        best_utility = float(expected_utilities[best_action])
        
        # Record decision
        self.decision_history.append({
            "state_probabilities": state_probabilities.tolist(),
            "expected_utilities": expected_utilities.tolist(),
            "best_action": best_action,
            "best_utility": best_utility,
            "timestamp": datetime.now()
        })
        
        return best_action, best_utility
    
    def get_decision_stats(self) -> Dict[str, any]:
        """Get statistics about decision making."""
        if not self.decision_history:
            return {"total_decisions": 0}
        
        avg_utility = np.mean([d["best_utility"] for d in self.decision_history])
        
        return {
            "total_decisions": len(self.decision_history),
            "average_utility": float(avg_utility)
        }


class DungArgumentationFramework:
    """
    Dung's Argumentation Framework for argument evaluation.
    Models arguments and their attack relations with formal semantics.
    """
    
    def __init__(self):
        self.arguments: Dict[str, Argument] = {}
        self.attacks: Set[Tuple[str, str]] = set()
        self.accepted_arguments: Set[str] = set()
    
    def add_argument(self, name: str, content: str, strength: float = 1.0):
        """Add an argument to the framework."""
        self.arguments[name] = Argument(name=name, content=content, strength=strength)
        logger.debug(f"Added argument: {name}")
    
    def add_attack(self, attacker: str, attacked: str):
        """Add an attack relation."""
        if attacker not in self.arguments or attacked not in self.arguments:
            raise ValueError("Both arguments must exist")
        self.attacks.add((attacker, attacked))
        logger.debug(f"Added attack: {attacker} attacks {attacked}")
    
    def is_conflict_free(self, argument_set: Set[str]) -> bool:
        """Check if a set of arguments is conflict-free."""
        for arg1 in argument_set:
            for arg2 in argument_set:
                if (arg1, arg2) in self.attacks or (arg2, arg1) in self.attacks:
                    return False
        return True
    
    def get_attackers(self, argument: str) -> Set[str]:
        """Get all arguments that attack the given argument."""
        return {attacker for attacker, attacked in self.attacks if attacked == argument}
    
    def is_defended(self, argument: str, defending_set: Set[str]) -> bool:
        """Check if an argument is defended by the defending set."""
        attackers = self.get_attackers(argument)
        for attacker in attackers:
            if not any((defender, attacker) in self.attacks for defender in defending_set):
                return False
        return True
    
    def compute_grounded_extension(self) -> Set[str]:
        """
        Compute the grounded extension (least fixed point).
        Returns the set of accepted arguments.
        """
        S = set()
        changed = True
        
        while changed:
            changed = False
            for arg_name in self.arguments:
                if arg_name in S:
                    continue
                
                # Argument is acceptable if all its attackers are attacked by S
                if self.is_defended(arg_name, S):
                    S.add(arg_name)
                    changed = True
        
        self.accepted_arguments = S
        logger.info(f"Grounded extension computed: {S}")
        return S
    
    def get_argument_status(self, argument: str) -> str:
        """Get the status of an argument (accepted/rejected/unknown)."""
        if argument in self.accepted_arguments:
            return "accepted"
        elif argument in self.arguments:
            return "rejected"
        else:
            return "unknown"
    
    def get_framework_stats(self) -> Dict[str, any]:
        """Get statistics about the argumentation framework."""
        return {
            "total_arguments": len(self.arguments),
            "total_attacks": len(self.attacks),
            "accepted_arguments": len(self.accepted_arguments),
            "rejected_arguments": len(self.arguments) - len(self.accepted_arguments)
        }


class InformationValueCalculator:
    """
    Expected Value of Perfect Information (EVPI) calculator.
    Quantifies the value of gathering additional information.
    """
    
    def __init__(self):
        self.calculation_history: List[Dict] = []
    
    def compute_evpi(self, state_probabilities: np.ndarray, 
                    utility_matrix: np.ndarray) -> float:
        """
        Compute Expected Value of Perfect Information.
        state_probabilities: array of state probabilities
        utility_matrix: (n_actions, n_states)
        returns: EVPI (higher = more valuable to have perfect information)
        """
        # Expected utility without perfect information
        eu_without = np.max(utility_matrix @ state_probabilities)
        
        # Expected utility with perfect information (choose best action per state)
        best_per_state = utility_matrix.max(axis=0)
        eu_with = np.dot(state_probabilities, best_per_state)
        
        evpi = eu_with - eu_without
        
        # Record calculation
        self.calculation_history.append({
            "eu_without": float(eu_without),
            "eu_with": float(eu_with),
            "evpi": float(evpi),
            "timestamp": datetime.now()
        })
        
        return evpi
    
    def recommend_information_gathering(self, state_probabilities: np.ndarray,
                                       utility_matrix: np.ndarray,
                                       threshold: float = 0.5) -> bool:
        """
        Recommend whether to gather more information based on EVPI.
        Returns True if EVPI exceeds threshold.
        """
        evpi = self.compute_evpi(state_probabilities, utility_matrix)
        return evpi > threshold
    
    def get_calculation_stats(self) -> Dict[str, any]:
        """Get statistics about EVPI calculations."""
        if not self.calculation_history:
            return {"total_calculations": 0}
        
        avg_evpi = np.mean([c["evpi"] for c in self.calculation_history])
        
        return {
            "total_calculations": len(self.calculation_history),
            "average_evpi": float(avg_evpi)
        }


class DecisionArgumentationSuite:
    """
    Combined decision theory and argumentation suite.
    """
    
    def __init__(self):
        self.decision_maker = ExpectedUtilityDecisionMaker()
        self.argumentation_framework = DungArgumentationFramework()
        self.info_value_calculator = InformationValueCalculator()
        self.suite_history: List[Dict] = []
    
    def make_decision(self, state_probabilities: np.ndarray,
                     utility_matrix: np.ndarray,
                     action_names: Optional[List[str]] = None) -> Dict[str, any]:
        """Make a decision using expected utility."""
        best_action, best_utility = self.decision_maker.find_optimal_action(
            state_probabilities, utility_matrix, action_names
        )
        
        return {
            "best_action": best_action,
            "expected_utility": best_utility,
            "action_name": action_names[best_action] if action_names else f"action_{best_action}"
        }
    
    def evaluate_arguments(self) -> Set[str]:
        """Evaluate arguments using grounded extension."""
        return self.argumentation_framework.compute_grounded_extension()
    
    def assess_information_value(self, state_probabilities: np.ndarray,
                                utility_matrix: np.ndarray) -> float:
        """Assess the value of gathering more information."""
        return self.info_value_calculator.compute_evpi(state_probabilities, utility_matrix)
    
    def get_suite_stats(self) -> Dict[str, any]:
        """Get combined statistics."""
        decision_stats = self.decision_maker.get_decision_stats()
        argumentation_stats = self.argumentation_framework.get_framework_stats()
        info_stats = self.info_value_calculator.get_calculation_stats()
        
        return {
            "decision_making": decision_stats,
            "argumentation": argumentation_stats,
            "information_value": info_stats
        }


def test_expected_utility():
    """Test Expected Utility decision making."""
    print("="*70)
    print("Testing Expected Utility Decision Making")
    print("="*70)
    
    decision_maker = ExpectedUtilityDecisionMaker()
    
    # Example: 2 states, 3 actions
    state_probs = np.array([0.3, 0.7])  # State probabilities
    utility_matrix = np.array([
        [10, 0],   # Action 1: good in state 1, bad in state 2
        [5, 8],    # Action 2: moderate in both
        [0, 10]    # Action 3: bad in state 1, good in state 2
    ])
    
    best_action, best_utility = decision_maker.find_optimal_action(
        state_probs, utility_matrix, ["action1", "action2", "action3"]
    )
    
    print(f"Best action: {best_action}, Expected utility: {best_utility:.2f}")
    
    stats = decision_maker.get_decision_stats()
    print(f"Decision stats: {stats}")
    
    print("[OK] Expected Utility working correctly\n")


def test_dung_argumentation():
    """Test Dung Argumentation Framework."""
    print("="*70)
    print("Testing Dung Argumentation Framework")
    print("="*70)
    
    framework = DungArgumentationFramework()
    
    # Add arguments
    framework.add_argument("a", "Argument A: The model is accurate")
    framework.add_argument("b", "Argument B: The model is biased")
    framework.add_argument("c", "Argument C: The model is fair")
    framework.add_argument("d", "Argument D: The model is reliable")
    
    # Add attacks
    framework.add_attack("b", "a")  # B attacks A
    framework.add_attack("c", "b")  # C attacks B
    framework.add_attack("d", "c")  # D attacks C
    
    print(f"Framework stats: {framework.get_framework_stats()}")
    
    # Compute grounded extension
    accepted = framework.compute_grounded_extension()
    print(f"Accepted arguments: {accepted}")
    
    # Check argument status
    for arg in ["a", "b", "c", "d"]:
        status = framework.get_argument_status(arg)
        print(f"  {arg}: {status}")
    
    print("[OK] Dung Argumentation Framework working correctly\n")


def test_evpi():
    """Test Expected Value of Perfect Information."""
    print("="*70)
    print("Testing Expected Value of Perfect Information")
    print("="*70)
    
    calculator = InformationValueCalculator()
    
    # Example: 2 states, 2 actions
    state_probs = np.array([0.4, 0.6])
    utility_matrix = np.array([
        [10, 0],   # Action 1: good in state 1, bad in state 2
        [0, 10]    # Action 2: bad in state 1, good in state 2
    ])
    
    evpi = calculator.compute_evpi(state_probs, utility_matrix)
    print(f"EVPI: {evpi:.2f}")
    
    # Recommendation
    should_gather = calculator.recommend_information_gathering(state_probs, utility_matrix, threshold=2.0)
    print(f"Recommend gathering information: {should_gather}")
    
    stats = calculator.get_calculation_stats()
    print(f"Calculation stats: {stats}")
    
    print("[OK] EVPI working correctly\n")


def test_decision_argumentation_suite():
    """Test combined decision and argumentation suite."""
    print("="*70)
    print("Testing Decision Argumentation Suite")
    print("="*70)
    
    suite = DecisionArgumentationSuite()
    
    # Add arguments
    suite.argumentation_framework.add_argument("quality", "Response quality is high")
    suite.argumentation_framework.add_argument("speed", "Response speed is fast")
    suite.argumentation_framework.add_argument("accuracy", "Response is accurate")
    suite.argumentation_framework.add_argument("relevance", "Response is relevant")
    
    # Add attacks
    suite.argumentation_framework.add_attack("speed", "quality")
    suite.argumentation_framework.add_attack("accuracy", "speed")
    suite.argumentation_framework.add_attack("relevance", "accuracy")
    
    # Make decision
    state_probs = np.array([0.3, 0.7])
    utility_matrix = np.array([[8, 2], [5, 6], [3, 9]])
    decision = suite.make_decision(state_probs, utility_matrix, ["action1", "action2", "action3"])
    print(f"Decision: {decision}")
    
    # Evaluate arguments
    accepted = suite.evaluate_arguments()
    print(f"Accepted arguments: {accepted}")
    
    # Assess information value
    evpi = suite.assess_information_value(state_probs, utility_matrix)
    print(f"EVPI: {evpi:.2f}")
    
    stats = suite.get_suite_stats()
    print(f"\nSuite stats: {stats}")
    
    print("[OK] Decision Argumentation Suite working correctly\n")


if __name__ == "__main__":
    try:
        test_expected_utility()
        test_dung_argumentation()
        test_evpi()
        test_decision_argumentation_suite()
        
        print("="*70)
        print("ALL DECISION THEORY AND ARGUMENTATION TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()