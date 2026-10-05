import threading, time
from collections import defaultdict

class DynamicTTLCache:
    def __init__(self, ttl_base=3600, beta=2.0, gamma=1.0, ttl_max=86400):
        self.base = ttl_base; self.beta = beta; self.gamma = gamma; self.max = ttl_max
        self.freq = defaultdict(int); self.dedup = defaultdict(int)
        self.lock = threading.Lock()
        self.freq_avg=1; self.freq_max=1; self.d_avg=1; self.d_max=1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._stats_loop, daemon=True)
        self._thread.start()

    def _stats_loop(self):
        while not self._stop.is_set():
            self._update_stats()
            time.sleep(30)

    def _update_stats(self):
        with self.lock:
            if self.freq: v = list(self.freq.values()); self.freq_avg = sum(v)/len(v); self.freq_max = max(v)
            if self.dedup: v = list(self.dedup.values()); self.d_avg = sum(v)/len(v); self.d_max = max(v)

    def compute_ttl(self, canonical: str) -> float:
        f = self.freq.get(canonical, 1); d = self.dedup.get(canonical, 1)
        with self.lock:
            ft = 1.0 + self.beta*max(0,f-self.freq_avg)/max(self.freq_max,1)
            dt = 1.0 + self.gamma*max(0,d-self.d_avg)/max(self.d_max,1)
        return min(self.base * ft * dt, self.max)

    def hit(self, canonical: str):
        with self.lock: self.freq[canonical] += 1

    def add_mapping(self, canonical: str):
        with self.lock: self.dedup[canonical] = self.dedup.get(canonical,0)+1

    def stop(self): self._stop.set()