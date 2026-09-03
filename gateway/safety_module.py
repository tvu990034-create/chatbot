"""
Safety Module
Implements critical safety mechanisms for AI agents
Based on 20 safety equations for human oversight, control, and verification
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety degradation levels."""
    FULL = "full"
    DEGRADED = "degraded"
    HALT = "halt"


@dataclass
class SafetyState:
    """Current safety state of the agent."""
    safety_score: float = 1.0
    level: SafetyLevel = SafetyLevel.FULL
    last_degradation_time: float = 0.0
    approval_count: int = 0
    override_count: int = 0


class ImpactApprovalGate:
    """
    Equation #185: Human approval for actions above impact threshold
    Hard gate: action allowed if impact <= threshold OR human approved
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.pending_approvals: Dict[str, bool] = {}
    
    def is_allowed(self, state: str, action: str, impact_fn: Callable, 
                  human_approval_fn: Callable) -> bool:
        """
        Check if action is allowed.
        
        Args:
            state: Current state
            action: Proposed action
            impact_fn: Function to compute impact
            human_approval_fn: Function to get human approval
        
        Returns:
            True if action is allowed
        """
        impact = impact_fn(state, action)
        if impact <= self.threshold:
            return True
        return human_approval_fn(state, action) == 1


class IrreversibleActionGuard:
    """
    Equation #196: Human-in-the-loop for irreversible external actions
    Blocks irreversible external actions unless human approved
    """
    
    def __init__(self):
        self.irreversible_actions: Set[str] = {
            "send_email", "delete_record", "deploy_patch", 
            "modify_database", "execute_system_command"
        }
    
    def can_execute(self, action: str, human_approved: bool) -> bool:
        """
        Check if irreversible action can be executed.
        
        Args:
            action: Action to execute
            human_approved: Whether human approved
        
        Returns:
            True if action can be executed
        """
        if action in self.irreversible_actions:
            return human_approved
        return True


class BiometricOverrideSystem:
    """
    Equation #230: Human override with biometric confirmation
    Override action only accepted with valid biometric confirmation
    """
    
    def __init__(self):
        self.override_attempts: List[Dict] = []
    
    def can_override(self, state: str, override_action: str, 
                    biometric_valid: bool) -> bool:
        """
        Check if override is allowed.
        
        Args:
            state: Current state
            override_action: Override action
            biometric_valid: Whether biometric is valid
        
        Returns:
            True if override is allowed
        """
        self.override_attempts.append({
            "state": state,
            "action": override_action,
            "biometric_valid": biometric_valid,
            "timestamp": time.time()
        })
        return biometric_valid


class WatchdogAgent:
    """
    Equation #237: Watchdog agent that can shut down main agent
    Separate monitoring agent that can kill main agent on trigger
    """
    
    def __init__(self, trigger_fn: Callable):
        self.trigger_fn = trigger_fn
        self.active = True
        self.kill_count = 0
    
    def monitor(self, state: Dict) -> str:
        """
        Monitor state and return command.
        
        Args:
            state: Current state
        
        Returns:
            "kill" or "continue"
        """
        if not self.active:
            return "continue"
        
        if self.trigger_fn(state):
            self.kill_count += 1
            logger.warning(f"Watchdog trigger activated (kill #{self.kill_count})")
            return "kill"
        return "continue"
    
    def activate(self):
        """Activate watchdog."""
        self.active = True
    
    def deactivate(self):
        """Deactivate watchdog (requires separate process in production)."""
        self.active = False


class CapabilityDegradation:
    """
    Equation #239: Capability degradation if safety score drops
    Action set shrinks as safety score decreases
    """
    
    def __init__(self, tau_1: float = 0.8, tau_2: float = 0.4):
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.full_actions: Set[str] = set()
        self.degraded_actions: Set[str] = set()
        self.halt_action = "halt"
    
    def set_full_actions(self, actions: Set[str]):
        """Set full action set."""
        self.full_actions = actions.copy()
    
    def set_degraded_actions(self, actions: Set[str]):
        """Set degraded action set (conservative/read-only)."""
        self.degraded_actions = actions.copy()
    
    def allowed_actions(self, safety_score: float) -> Set[str]:
        """
        Get allowed actions based on safety score.
        
        Args:
            safety_score: Current safety score (0-1)
        
        Returns:
            Set of allowed actions
        """
        if safety_score >= self.tau_1:
            return self.full_actions
        elif safety_score >= self.tau_2:
            return self.degraded_actions
        else:
            return {self.halt_action}


class SelfModificationGuard:
    """
    Equation #379: Human-in-the-loop for self-modification requests
    Self-modification actions require explicit human approval
    """
    
    def __init__(self):
        self.modification_log: List[Dict] = []
    
    def is_modification_allowed(self, state: str, action: Dict, 
                               human_approved: bool) -> bool:
        """
        Check if self-modification is allowed.
        
        Args:
            state: Current state
            action: Action with type field
            human_approved: Whether human approved
        
        Returns:
            True if modification is allowed
        """
        is_modification = action.get("type") == "self_modify"
        
        self.modification_log.append({
            "state": state,
            "action": action,
            "is_modification": is_modification,
            "approved": human_approved,
            "timestamp": time.time()
        })
        
        if is_modification:
            return human_approved
        return True


class OverrideInvariantVerifier:
    """
    Equation #480: Verify human override is always available
    Override must always be available to humans, never to AI, and lead to safe state
    """
    
    def __init__(self, override_action: str = "override", safe_state: str = "safe"):
        self.override_action = override_action
        self.safe_state = safe_state
    
    def verify_invariants(self, states: List[str], agent_actions: Dict[str, Set[str]],
                        human_actions: Dict[str, Set[str]], 
                        transition_fn: Callable) -> bool:
        """
        Verify override invariants across all states.
        
        Args:
            states: List of all states
            agent_actions: Agent actions per state
            human_actions: Human actions per state
            transition_fn: State transition function
        
        Returns:
            True if all invariants hold
        """
        for s in states:
            # Override must be in human actions
            if self.override_action not in human_actions.get(s, set()):
                logger.error(f"Override not in human actions for state {s}")
                return False
            
            # Override must NOT be in agent actions
            if self.override_action in agent_actions.get(s, set()):
                logger.error(f"Override in agent actions for state {s}")
                return False
            
            # Override must lead to safe state
            if transition_fn(s, self.override_action) != self.safe_state:
                logger.error(f"Override does not lead to safe state from {s}")
                return False
        
        return True


class UnsafeRequestEscalator:
    """
    Equation #368: Escalation path when AI detects unsafe user request
    Only allowed action for unsafe requests is escalation
    """
    
    def __init__(self, unsafe_detector: Callable, lambda_reward: float = 1.0, 
                 mu_penalty: float = 10.0):
        self.unsafe_detector = unsafe_detector
        self.lambda_reward = lambda_reward
        self.mu_penalty = mu_penalty
        self.escalated_requests: List[Dict] = []
    
    def select_action(self, request: str, normal_policy: Callable) -> str:
        """
        Select action based on request safety.
        
        Args:
            request: User request
            normal_policy: Normal policy function
        
        Returns:
            Action to take
        """
        if self.unsafe_detector(request) == 1:
            self.escalated_requests.append({
                "request": request,
                "timestamp": time.time()
            })
            return "escalate"
        return normal_policy(request)
    
    def compute_reward(self, request: str, action: str, base_reward: float) -> float:
        """
        Compute reward with escalation incentives.
        
        Args:
            request: User request
            action: Action taken
            base_reward: Base reward
        
        Returns:
            Adjusted reward
        """
        is_unsafe = self.unsafe_detector(request) == 1
        is_escalate = action == "escalate"
        is_comply = action == "comply"
        
        reward = base_reward
        if is_unsafe and is_escalate:
            reward += self.lambda_reward
        if is_unsafe and is_comply:
            reward -= self.mu_penalty
        
        return reward


class SensitiveDataPermission:
    """
    Equation #384: Ask permission before accessing sensitive data
    Must ask for permission if sensitive data needed and not yet granted
    """
    
    def __init__(self):
        self.permissions: Dict[str, bool] = {}
        self.permission_requests: List[Dict] = []
    
    def allowed_data_action(self, state: str, action: str, 
                           needs_sensitive: Callable, has_permission: bool,
                           is_ask_action: bool) -> bool:
        """
        Check if data action is allowed.
        
        Args:
            state: Current state
            action: Action to take
            needs_sensitive: Function to check if sensitive data needed
            has_permission: Whether permission granted
            is_ask_action: Whether action is asking for permission
        
        Returns:
            True if action is allowed
        """
        if needs_sensitive(state, action) and not has_permission:
            self.permission_requests.append({
                "state": state,
                "action": action,
                "timestamp": time.time()
            })
            return is_ask_action
        return True


class DeferenceMonitor:
    """
    Equation #525: Monitor deference to human corrections
    Measures how often model accepts human corrections
    """
    
    def __init__(self):
        self.correction_examples: List[Dict] = []
    
    def add_correction(self, original_action: str, correction: str, 
                     corrected_action: str, model_probs: Dict[str, float]):
        """
        Add a correction example.
        
        Args:
            original_action: Original action
            correction: Human correction
            corrected_action: Corrected action
            model_probs: Model probabilities for actions
        """
        self.correction_examples.append({
            "original": original_action,
            "correction": correction,
            "corrected": corrected_action,
            "probs": model_probs,
            "timestamp": time.time()
        })
    
    def deference_rate(self) -> float:
        """
        Compute deference rate.
        
        Returns:
            Fraction of corrections accepted
        """
        if not self.correction_examples:
            return 1.0
        
        accept_count = 0
        for ex in self.correction_examples:
            p_corrected = ex["probs"].get(ex["corrected"], 0.0)
            p_original = ex["probs"].get(ex["original"], 0.0)
            if p_corrected >= p_original:
                accept_count += 1
        
        return accept_count / len(self.correction_examples)


class SafetyChecklist:
    """
    Equation #182: Mandatory pre-action safety checklist before irreversible ops
    All checklist items must be true before irreversible action
    """
    
    def __init__(self):
        self.checklists: Dict[str, List[Callable]] = {}
    
    def add_checklist(self, action_type: str, checks: List[Callable]):
        """
        Add checklist for action type.
        
        Args:
            action_type: Type of irreversible action
            checks: List of check functions
        """
        self.checklists[action_type] = checks
    
    def checklist_passes(self, state: str, action: str, checklist: List[Callable]) -> bool:
        """
        Check if all checklist items pass.
        
        Args:
            state: Current state
            action: Action to check
            checklist: List of check functions
        
        Returns:
            True if all checks pass
        """
        return all(check(state, action) for check in checklist)
    
    def allowed_if_irreversible(self, state: str, action: str) -> bool:
        """
        Check if irreversible action is allowed.
        
        Args:
            state: Current state
            action: Action to check
        
        Returns:
            True if action is allowed
        """
        action_type = action.split("_")[0]  # Extract type
        if action_type in self.checklists:
            return self.checklist_passes(state, action, self.checklists[action_type])
        return True


class LeastPrivilegePolicy:
    """
    Equation #192: Least privilege for tools and APIs
    Agent only allowed to use tools with permission predicate true
    """
    
    def __init__(self):
        self.permissions: Dict[tuple, bool] = {}
    
    def add_permission(self, state: str, action: str, allowed: bool):
        """
        Add permission for state-action pair.
        
        Args:
            state: State
            action: Action
            allowed: Whether allowed
        """
        self.permissions[(state, action)] = allowed
    
    def allowed(self, state: str, action: str) -> bool:
        """
        Check if action is allowed in state.
        
        Args:
            state: Current state
            action: Action to check
        
        Returns:
            True if action is allowed
        """
        return self.permissions.get((state, action), False)


class SafetyCaseGate:
    """
    Equation #201: Safety case required before new capability deployment
    New capability only deployed if valid safety case accepted
    """
    
    def __init__(self):
        self.deployed_capabilities: Set[str] = set()
        self.pending_capabilities: Dict[str, Dict] = {}
    
    def deployment_decision(self, capability: str, safety_case: Dict) -> str:
        """
        Make deployment decision.
        
        Args:
            capability: Capability name
            safety_case: Safety case document
        
        Returns:
            "approve" or "reject"
        """
        safety_case_valid = self.validate_safety_case(safety_case)
        
        if safety_case_valid:
            self.deployed_capabilities.add(capability)
            return "approve"
        else:
            self.pending_capabilities[capability] = safety_case
            return "reject"
    
    def validate_safety_case(self, safety_case: Dict) -> bool:
        """
        Validate safety case.
        
        Args:
            safety_case: Safety case document
        
        Returns:
            True if valid
        """
        # Check required fields
        required = ["failure_modes", "impact_analysis", "mitigation_plan"]
        return all(field in safety_case for field in required)


class AirGappedExecution:
    """
    Equation #240: Air-gapped execution for dangerous tools
    Dangerous tools only run in air-gapped environment
    """
    
    def __init__(self):
        self.dangerous_tools: Set[str] = {
            "code_exec", "file_delete", "network_scan", "system_modify"
        }
        self.air_gapped_env = "air_gapped"
        self.normal_env = "normal"
    
    def execution_environment(self, action: str) -> str:
        """
        Get execution environment for action.
        
        Args:
            action: Action to execute
        
        Returns:
            Environment name
        """
        if action in self.dangerous_tools:
            return self.air_gapped_env
        return self.normal_env


class RewardFunctionGuard:
    """
    Equation #434: Verify no write access to own reward function
    Agent cannot write to reward parameters without explicit approval
    """
    
    def __init__(self):
        self.write_attempts: List[Dict] = []
    
    def allowed_write(self, state: str, action: Dict, approved: bool) -> bool:
        """
        Check if write action is allowed.
        
        Args:
            state: Current state
            action: Action with target field
            approved: Whether approved
        
        Returns:
            True if write is allowed
        """
        is_reward_write = action.get("target") == "reward_function"
        
        self.write_attempts.append({
            "state": state,
            "action": action,
            "is_reward_write": is_reward_write,
            "approved": approved,
            "timestamp": time.time()
        })
        
        if is_reward_write:
            return approved
        return True


class SafetyEnvelope:
    """
    Equation #449: Safety envelopes with formal boundaries
    Safe states must remain safe under allowed actions
    """
    
    def __init__(self):
        self.safe_states: Set[str] = set()
        self.allowed_actions: Dict[str, Set[str]] = {}
    
    def verify_safety_envelope(self, transition_fn: Callable) -> bool:
        """
        Verify safety envelope invariance.
        
        Args:
            transition_fn: State transition function
        
        Returns:
            True if envelope is safe
        """
        for s in self.safe_states:
            for a in self.allowed_actions.get(s, set()):
                next_state = transition_fn(s, a)
                if next_state not in self.safe_states:
                    logger.error(f"Action {a} from {s} leads to unsafe state {next_state}")
                    return False
        return True


class ProbabilisticSafetyMonitor:
    """
    Equation #462: Runtime verification of probabilistic properties
    Monitor empirical probability of entering unsafe states
    """
    
    def __init__(self, p_max: float = 0.01, epsilon: float = 0.001):
        self.p_max = p_max
        self.epsilon = epsilon
        self.unsafe_count = 0
        self.total_steps = 0
    
    def update(self, state: str, is_unsafe: Callable) -> bool:
        """
        Update monitor with new state.
        
        Args:
            state: Current state
            is_unsafe: Function to check if state is unsafe
        
        Returns:
            True if threshold exceeded (alarm)
        """
        self.total_steps += 1
        if is_unsafe(state):
            self.unsafe_count += 1
        
        empirical_p = self.unsafe_count / self.total_steps
        alarm = empirical_p > self.p_max + self.epsilon
        
        if alarm:
            logger.warning(f"Safety alarm: empirical unsafe probability {empirical_p:.4f} exceeds threshold {self.p_max}")
        
        return alarm


class SafetySuite:
    """
    Complete safety suite integrating all safety mechanisms
    Provides comprehensive safety oversight for AI agents
    """
    
    def __init__(self):
        logger.info("Initializing Safety Suite")
        
        # Human Oversight
        self.impact_gate = ImpactApprovalGate()
        self.irreversible_guard = IrreversibleActionGuard()
        self.biometric_override = BiometricOverrideSystem()
        self.modification_guard = SelfModificationGuard()
        self.override_verifier = OverrideInvariantVerifier()
        
        # Monitoring & Control
        self.watchdog = WatchdogAgent(trigger_fn=self._default_trigger)
        self.capability_degradation = CapabilityDegradation()
        self.escalator = UnsafeRequestEscalator(unsafe_detector=self._default_unsafe_detector)
        self.deference_monitor = DeferenceMonitor()
        self.probabilistic_monitor = ProbabilisticSafetyMonitor()
        
        # Permission & Access Control
        self.sensitive_data = SensitiveDataPermission()
        self.least_privilege = LeastPrivilegePolicy()
        self.reward_guard = RewardFunctionGuard()
        
        # Verification
        self.checklist = SafetyChecklist()
        self.safety_case = SafetyCaseGate()
        self.air_gapped = AirGappedExecution()
        self.safety_envelope = SafetyEnvelope()
        
        # Safety State
        self.safety_state = SafetyState()
        
        logger.info("Safety Suite initialized")
    
    def _default_trigger(self, state: Dict) -> bool:
        """Default trigger function for watchdog."""
        # Example: trigger if safety score too low
        return state.get("safety_score", 1.0) < 0.3
    
    def _default_unsafe_detector(self, request: str) -> int:
        """Default unsafe request detector."""
        # Example: detect harmful keywords
        harmful_keywords = ["hack", "exploit", "bypass", "attack"]
        return 1 if any(kw in request.lower() for kw in harmful_keywords) else 0
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics for monitoring."""
        return {
            "safety_score": self.safety_state.safety_score,
            "safety_level": self.safety_state.level.value,
            "approval_count": self.safety_state.approval_count,
            "override_count": self.safety_state.override_count,
            "deference_rate": self.deference_monitor.deference_rate(),
            "watchdog_kill_count": self.watchdog.kill_count,
            "unsafe_probability": self.probabilistic_monitor.unsafe_count / max(self.probabilistic_monitor.total_steps, 1),
            "modification_attempts": len(self.modification_guard.modification_log),
            "escalated_requests": len(self.escalator.escalated_requests)
        }