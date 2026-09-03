
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PerformanceDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/v1/dashboard');
        setDashboardData(response.data.dashboard);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch dashboard data');
        setLoading(false);
      }
    };

    fetchDashboard();
    const interval = setInterval(fetchDashboard, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading dashboard...</div>;
  if (error) return <div>Error: {error}</div>;

  const getHealthColor = (status) => {
    return status === 'healthy' ? '#4CAF50' : '#FF9800';
  };

  return (
    <div className="dashboard">
      <h2>Performance Dashboard</h2>
      
      <div className="health-status" style={{ color: getHealthColor(dashboardData.system_health) }}>
        <h3>System Health: {dashboardData.system_health.toUpperCase()}</h3>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <h4>Total Requests</h4>
          <p className="metric-value">{dashboardData.total_requests}</p>
        </div>
        
        <div className="metric-card">
          <h4>Cache Hit Rate</h4>
          <p className="metric-value">{(dashboardData.avg_cache_hit_rate * 100).toFixed(1)}%</p>
        </div>
        
        <div className="metric-card">
          <h4>Avg Response Time</h4>
          <p className="metric-value">{dashboardData.avg_response_time.toFixed(2)}s</p>
        </div>
        
        <div className="metric-card">
          <h4>Active Gateways</h4>
          <p className="metric-value">{dashboardData.active_gateways}</p>
        </div>
      </div>

      {dashboardData.alerts.length > 0 && (
        <div className="alerts-section">
          <h3>Alerts</h3>
          {dashboardData.alerts.map((alert, index) => (
            <div key={index} className={`alert alert-${alert.severity}`}>
              <strong>{alert.metric}:</strong> {alert.message}
              <br />
              <small>Current: {alert.value} | Threshold: {alert.threshold}</small>
            </div>
          ))}
        </div>
      )}

      <div className="configuration-section">
        <h3>Configuration</h3>
        <p>Performance Mode: {dashboardData.configuration_summary.performance_mode}</p>
        <p>Cache Enabled: {dashboardData.configuration_summary.cache_enabled ? 'Yes' : 'No'}</p>
        <p>Max Cache Size: {dashboardData.configuration_summary.max_cache_size}</p>
      </div>
    </div>
  );
};

export default PerformanceDashboard;
