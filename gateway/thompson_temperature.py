"""
Thompson Sampling for Adaptive Temperature
Bayesian bandit algorithm that maintains Beta posterior over temperature effectiveness
Balances exploration and exploitation for optimal temperature selection
"""

import numpy as np
from collections import deque

class ThompsonSamplingTemperature:
    """
    Thompson Sampling for adaptive temperature selection.
    Maintains Beta posterior over discrete temperature values.
    Samples from posterior to select temperature, updates based on success/failure.
    """

    def __init__(self, temperature_options=None, window_size=20):
        """
        Initialize with discrete temperature options.
        If None, uses [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        """
        if temperature_options is None:
            self.temperature_options = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
        else:
            self.temperature_options = np.array(temperature_options)

        self.num_options = len(self.temperature_options)
        # Beta priors: alpha (successes), beta (failures)
        self.alpha = np.ones(self.num_options)  # Prior: uniform
        self.beta = np.ones(self.num_options)

        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.current_temp_idx = None

    def select_temperature(self):
        """
        Select temperature using Thompson Sampling.
        Sample from Beta posterior for each temperature, choose max.
        """
        # Sample from Beta posterior for each temperature option
        samples = np.random.beta(self.alpha, self.beta)
        self.current_temp_idx = np.argmax(samples)
        return self.temperature_options[self.current_temp_idx]

    def record_feedback(self, success: bool):
        """
        Record feedback for the selected temperature.
        Updates Beta posterior for the chosen temperature.
        """
        if self.current_temp_idx is None:
            return

        # Update Beta parameters for the selected temperature
        if success:
            self.alpha[self.current_temp_idx] += 1
        else:
            self.beta[self.current_temp_idx] += 1

        # Record in history
        self.history.append((self.temperature_options[self.current_temp_idx], success))

    def get_temperature(self):
        """Get current temperature (selects using Thompson Sampling)."""
        return self.select_temperature()

    def get_statistics(self):
        """Return statistics about temperature selection."""
        if len(self.history) == 0:
            return {
                "total_selections": 0,
                "temperature_distribution": {float(t): 0 for t in self.temperature_options},
                "success_rates": {float(t): 0.0 for t in self.temperature_options}
            }

        # Count selections
        selections = np.zeros(self.num_options)
        successes = np.zeros(self.num_options)

        for temp, success in self.history:
            idx = np.where(self.temperature_options == temp)[0][0]
            selections[idx] += 1
            if success:
                successes[idx] += 1

        # Calculate success rates
        success_rates = np.divide(
            successes,
            selections,
            out=np.zeros_like(successes),
            where=selections != 0
        )

        return {
            "total_selections": len(self.history),
            "temperature_distribution": {
                float(t): int(s) for t, s in zip(self.temperature_options, selections)
            },
            "success_rates": {
                float(t): float(sr) for t, sr in zip(self.temperature_options, success_rates)
            },
            "current_posterior_means": {
                float(t): float(a / (a + b))
                for t, a, b in zip(self.temperature_options, self.alpha, self.beta)
            }
        }

    def reset(self):
        """Reset to initial state."""
        self.alpha = np.ones(self.num_options)
        self.beta = np.ones(self.num_options)
        self.history.clear()
        self.current_temp_idx = None


# Global instance
_thompson_temp = ThompsonSamplingTemperature()


def get_thompson_temperature():
    """Get the global Thompson Sampling temperature instance."""
    return _thompson_temp


def record_thompson_feedback(success: bool):
    """Record feedback for the Thompson Sampling temperature system."""
    _thompson_temp.record_feedback(success)
