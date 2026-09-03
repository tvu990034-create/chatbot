"""
Lightweight Online Gradient Descent for Adaptive Temperature
Implements simple online learning to adjust temperature based on feedback
"""

import time
from collections import deque

class AdaptiveTemperature:
    """
    Simple online gradient descent for temperature adjustment.
    Updates temperature based on recent response quality to maintain optimal calibration.
    """

    def __init__(self, initial_temp=0.7, learning_rate=0.01, window_size=10):
        self.temperature = initial_temp
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.history = deque(maxlen=window_size)  # Store recent (temperature, success) pairs
        self.min_temp = 0.1
        self.max_temp = 1.5

    def record_feedback(self, success: bool):
        """
        Record feedback on whether the last response was good (success=True) or bad (success=False).
        This is a simple heuristic - in practice, you might want more sophisticated feedback.
        """
        self.history.append((self.temperature, 1.0 if success else 0.0))

    def update_temperature(self):
        """
        Update temperature using online gradient descent.
        If recent success rate is high, we might want to be more conservative (lower temp).
        If recent success rate is low, we might want to be more exploratory (higher temp).
        """
        if len(self.history) < 3:
            return self.temperature  # Not enough data yet

        # Calculate recent success rate
        recent_temps = [t for t, _ in self.history]
        recent_success = [s for _, s in self.history]
        avg_success = sum(recent_success) / len(recent_success)

        # Simple gradient: if success is high, reduce temp slightly; if low, increase temp
        # This is a heuristic - could be more sophisticated
        if avg_success > 0.8:
            gradient = 0.01  # Push towards lower temp
        elif avg_success < 0.5:
            gradient = -0.01  # Push towards higher temp
        else:
            gradient = 0.0  # Stay where we are

        # Update temperature
        self.temperature = self.temperature - self.learning_rate * gradient

        # Clamp to valid range
        self.temperature = max(self.min_temp, min(self.max_temp, self.temperature))

        return self.temperature

    def get_temperature(self):
        """Get current temperature."""
        return self.temperature

    def reset(self):
        """Reset to initial state."""
        self.temperature = 0.7
        self.history.clear()


# Global instance
_adaptive_temp = AdaptiveTemperature()


def get_adaptive_temperature():
    """Get the global adaptive temperature instance."""
    return _adaptive_temp


def record_response_feedback(success: bool):
    """Record feedback for the adaptive temperature system."""
    _adaptive_temp.record_feedback(success)
    _adaptive_temp.update_temperature()
