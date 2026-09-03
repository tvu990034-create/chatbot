# Time-Series Data Storage for Analytics Enhancement
# Implementing historical performance data tracking

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

class TimeSeriesStorage:
    """Time-series data storage for performance analytics."""
    
    def __init__(self, storage_dir: str = "analytics_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metrics_file = self.storage_dir / "metrics_history.json"
        self.max_data_points = 1000  # Maximum data points to store per metric
    
    def store_metric(self, metric_name: str, value: float, timestamp: float = None):
        """Store a metric value with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        # Load existing data
        data = self._load_data()
        
        # Add new data point
        if metric_name not in data:
            data[metric_name] = []
        
        data[metric_name].append({
            "timestamp": timestamp,
            "value": value,
            "datetime": datetime.fromtimestamp(timestamp).isoformat()
        })
        
        # Trim to max data points
        if len(data[metric_name]) > self.max_data_points:
            data[metric_name] = data[metric_name][-self.max_data_points:]
        
        # Save data
        self._save_data(data)
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for specified time period."""
        data = self._load_data()
        
        if metric_name not in data:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            point for point in data[metric_name]
            if point["timestamp"] >= cutoff_time
        ]
    
    def get_aggregated_metrics(self, metric_name: str, interval: str = "hour") -> Dict[str, float]:
        """Get aggregated metrics over time intervals."""
        history = self.get_metric_history(metric_name, hours=24)
        
        if not history:
            return {}
        
        # Simple aggregation
        values = [point["value"] for point in history]
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
            "latest": values[-1] if values else 0
        }
    
    def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load metrics data from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_data(self, data: Dict[str, List[Dict[str, Any]]]):
        """Save metrics data to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

class PerformanceAlerting:
    """Performance alerting system."""
    
    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage
        self.alert_rules = {
            "response_time": {"threshold": 5.0, "comparison": "greater"},
            "cache_hit_rate": {"threshold": 0.3, "comparison": "less"},
            "error_rate": {"threshold": 0.05, "comparison": "greater"}
        }
        self.alert_history = []
    
    def check_thresholds(self, current_metrics: Dict[str, float]) -> List[Dict[str, str]]:
        """Check if current metrics exceed thresholds."""
        alerts = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.alert_rules:
                rule = self.alert_rules[metric_name]
                threshold = rule["threshold"]
                comparison = rule["comparison"]
                
                triggered = False
                if comparison == "greater" and current_value > threshold:
                    triggered = True
                elif comparison == "less" and current_value < threshold:
                    triggered = True
                
                if triggered:
                    alert = {
                        "metric": metric_name,
                        "current_value": current_value,
                        "threshold": threshold,
                        "comparison": comparison,
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning" if self._get_severity(metric_name, current_value, threshold) == "warning" else "critical"
                    }
                    alerts.append(alert)
                    self.alert_history.append(alert)
        
        return alerts
    
    def _get_severity(self, metric_name: str, current_value: float, threshold: float) -> str:
        """Determine alert severity based on how far threshold is exceeded."""
        if metric_name == "response_time":
            return "critical" if current_value > threshold * 2 else "warning"
        elif metric_name == "cache_hit_rate":
            return "critical" if current_value < threshold * 0.5 else "warning"
        else:
            return "warning"
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, str]]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
        ]

# Usage example (integrated with analytics endpoint)
if __name__ == "__main__":
    storage = TimeSeriesStorage()
    alerting = PerformanceAlerting(storage)
    
    # Store some sample metrics
    storage.store_metric("response_time", 2.5)
    storage.store_metric("cache_hit_rate", 0.45)
    storage.store_metric("error_rate", 0.01)
    
    # Check thresholds
    current_metrics = {
        "response_time": 2.5,
        "cache_hit_rate": 0.45,
        "error_rate": 0.01
    }
    
    alerts = alerting.check_thresholds(current_metrics)
    print(f"Alerts: {alerts}")
    
    # Get historical data
    history = storage.get_metric_history("response_time", hours=1)
    print(f"Response time history: {len(history)} data points")