import numpy as np
from scipy.optimize import curve_fit
from collections import deque

class ExponentialCacheTuner:
    def __init__(self, target_hit=0.9, max_threshold=8):
        self.target = target_hit
        self.max_th = max_threshold
        self.history = deque(maxlen=100)
        self.threshold = 2
        self.H0 = 0.4
        self.tau = 2.0

    def record(self, hits, total):
        if total: self.history.append((self.threshold, hits/total, total))
        if len(self.history) < 20: return
        ths = np.array([h[0] for h in self.history])
        hrs = np.array([h[1] for h in self.history])
        def model(d, H0, tau): return H0 * np.exp(-d/tau)
        try:
            popt, _ = curve_fit(model, ths, hrs, p0=[self.H0, self.tau], bounds=(0,[1,10]))
            self.H0, self.tau = popt
        except: pass

    def update(self) -> int:
        if self.H0 and self.target < self.H0:
            delta = -self.tau * np.log(self.target/self.H0)
            self.threshold = int(round(np.clip(delta, 1, self.max_th)))
        return self.threshold
    