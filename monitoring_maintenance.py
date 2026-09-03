"""
Automated Monitoring and Maintenance System
Provides health checks, performance monitoring, and automated maintenance tasks
"""

import requests
import time
import json
import subprocess
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SystemMonitor:
    """Comprehensive system monitoring and maintenance."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.monitoring_log = Path("monitoring_logs")
        self.monitoring_log.mkdir(exist_ok=True)
        self.alert_thresholds = {
            "response_time": 5.0,
            "cache_hit_rate": 0.3,
            "error_rate": 0.05,
            "memory_usage": 0.8  # 80% memory usage
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # API Health
        try:
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
            health_status["checks"]["api"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status["checks"]["api"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Analytics Health
        try:
            response = requests.get(f"{self.api_url}/api/v1/analytics", timeout=5)
            if response.status_code == 200:
                analytics = response.json()
                gateway_data = list(analytics.get("analytics", {}).values())[0] if analytics.get("analytics") else {}
                health_status["checks"]["analytics"] = {
                    "status": "healthy",
                    "cache_hit_rate": gateway_data.get("cache_hit_rate", 0),
                    "avg_response_time": gateway_data.get("avg_response_time", 0)
                }
            else:
                health_status["checks"]["analytics"] = {"status": "unhealthy"}
        except Exception as e:
            health_status["checks"]["analytics"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Dashboard Health
        try:
            response = requests.get(f"{self.api_url}/api/v1/dashboard", timeout=5)
            if response.status_code == 200:
                dashboard = response.json()
                health_status["checks"]["dashboard"] = {
                    "status": dashboard.get("dashboard", {}).get("system_health", "unknown")
                }
            else:
                health_status["checks"]["dashboard"] = {"status": "unhealthy"}
        except Exception as e:
            health_status["checks"]["dashboard"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall Health
        all_healthy = all(
            check.get("status") == "healthy" 
            for check in health_status["checks"].values()
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    
    def performance_monitoring(self) -> Dict[str, Any]:
        """Monitor system performance metrics."""
        try:
            response = requests.get(f"{self.api_url}/api/v1/dashboard", timeout=5)
            dashboard = response.json()["dashboard"]
            
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "total_requests": dashboard["total_requests"],
                "avg_cache_hit_rate": dashboard["avg_cache_hit_rate"],
                "avg_response_time": dashboard["avg_response_time"],
                "system_health": dashboard["system_health"],
                "active_gateways": dashboard["active_gateways"],
                "alerts": dashboard.get("alerts", [])
            }
            
            # Check against thresholds
            performance_data["threshold_checks"] = self._check_thresholds(performance_data)
            
            return performance_data
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "monitoring_failed"
            }
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Public method to get performance data."""
        return self.performance_monitoring()
    
    def _check_thresholds(self, performance_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check performance against thresholds."""
        threshold_breaches = []
        
        if performance_data.get("avg_response_time", 0) > self.alert_thresholds["response_time"]:
            threshold_breaches.append({
                "metric": "response_time",
                "current": performance_data["avg_response_time"],
                "threshold": self.alert_thresholds["response_time"],
                "severity": "warning"
            })
        
        if performance_data.get("avg_cache_hit_rate", 1.0) < self.alert_thresholds["cache_hit_rate"]:
            threshold_breaches.append({
                "metric": "cache_hit_rate",
                "current": performance_data["avg_cache_hit_rate"],
                "threshold": self.alert_thresholds["cache_hit_rate"],
                "severity": "warning"
            })
        
        return threshold_breaches
    
    def log_monitoring_data(self, data: Dict[str, Any]):
        """Log monitoring data to file."""
        log_file = self.monitoring_log / f"monitoring_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + "\n")
    
    def automated_cleanup(self):
        """Perform automated cleanup tasks."""
        cleanup_results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": []
        }
        
        # Clean old monitoring logs (keep last 7 days)
        self._clean_old_logs(cleanup_results)
        
        # Clean old analytics data (if applicable)
        self._clean_old_analytics(cleanup_results)
        
        # Clean old ML data (if applicable)
        self._clean_old_ml_data(cleanup_results)
        
        return cleanup_results
    
    def _clean_old_logs(self, results: Dict[str, Any]):
        """Clean old monitoring logs."""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in self.monitoring_log.glob("monitoring_*.jsonl"):
            try:
                file_date = datetime.strptime(log_file.stem.replace("monitoring_", ""), "%Y%m%d")
                if file_date < cutoff_date:
                    log_file.unlink()
                    results["tasks"].append({
                        "task": "deleted_old_log",
                        "file": str(log_file),
                        "status": "success"
                    })
            except:
                pass
    
    def _clean_old_analytics(self, results: Dict[str, Any]):
        """Clean old analytics data."""
        analytics_dir = Path("analytics_data")
        if analytics_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for data_file in analytics_dir.glob("*.json"):
                try:
                    file_date = datetime.fromtimestamp(data_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        data_file.unlink()
                        results["tasks"].append({
                            "task": "deleted_old_analytics",
                            "file": str(data_file),
                            "status": "success"
                        })
                except:
                    pass
    
    def _clean_old_ml_data(self, results: Dict[str, Any]):
        """Clean old ML training data."""
        ml_data_dir = Path("ml_data")
        if ml_data_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for data_file in ml_data_dir.glob("*.json"):
                try:
                    file_date = datetime.fromtimestamp(data_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        data_file.unlink()
                        results["tasks"].append({
                            "task": "deleted_old_ml_data",
                            "file": str(data_file),
                            "status": "success"
                        })
                except:
                    pass
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert notification (placeholder for email/Slack integration)."""
        print(f"ALERT: {alert_data}")
        # Log alert
        alert_file = self.monitoring_log / "alerts.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "alert": alert_data
            }) + "\n")

class MaintenanceScheduler:
    """Schedule and execute maintenance tasks."""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.last_health_check = None
        self.last_cleanup = None
    
    def run_daily_tasks(self):
        """Run daily maintenance tasks."""
        print("=== Running Daily Maintenance Tasks ===")
        
        # Health check
        health_status = self.monitor.health_check()
        self.monitor.log_monitoring_data({
            "task": "daily_health_check",
            "result": health_status
        })
        
        # Performance monitoring
        performance_data = self.monitor.performance_monitoring()
        self.monitor.log_monitoring_data({
            "task": "daily_performance_monitoring",
            "result": performance_data
        })
        
        # Check for alerts
        if performance_data.get("threshold_checks"):
            for alert in performance_data["threshold_checks"]:
                self.monitor.send_alert(alert)
        
        self.last_health_check = datetime.now()
        print("✅ Daily maintenance tasks completed")
    
    def run_weekly_tasks(self):
        """Run weekly maintenance tasks."""
        print("=== Running Weekly Maintenance Tasks ===")
        
        # Automated cleanup
        cleanup_results = self.monitor.automated_cleanup()
        self.monitor.log_monitoring_data({
            "task": "weekly_cleanup",
            "result": cleanup_results
        })
        
        self.last_cleanup = datetime.now()
        print("✅ Weekly maintenance tasks completed")
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "health_status": self.monitor.health_check(),
            "performance_data": self.monitor.performance_monitoring(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
            "system_info": self._get_system_info()
        }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": "unknown",
            "memory_total": "unknown",
            "memory_available": "unknown",
            "disk_usage": "unknown"
        }
        
        # Try to get detailed system info if psutil is available
        try:
            import psutil
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent
            })
        except ImportError:
            pass  # psutil not available, use basic info
        
        return system_info

# Main monitoring loop
def main():
    """Main monitoring and maintenance loop."""
    print("=" * 60)
    print("MONITORING AND MAINTENANCE SYSTEM")
    print("=" * 60)
    
    scheduler = MaintenanceScheduler()
    
    print("\nRunning comprehensive health check...")
    health_status = scheduler.monitor.health_check()
    print(f"Overall Health: {health_status['overall_status']}")
    
    print("\nRunning performance monitoring...")
    performance_data = scheduler.monitor.performance_monitoring()
    print(f"Total Requests: {performance_data.get('total_requests', 0)}")
    print(f"Cache Hit Rate: {performance_data.get('avg_cache_hit_rate', 0):.1%}")
    print(f"Response Time: {performance_data.get('avg_response_time', 0):.2f}s")
    
    print("\nGenerating maintenance report...")
    report = scheduler.generate_maintenance_report()
    
    # Save report
    report_file = Path("monitoring_logs") / f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Maintenance report saved to: {report_file}")
    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()