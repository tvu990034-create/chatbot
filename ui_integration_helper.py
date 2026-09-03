"""
UI Integration Helper - React Components for Enhanced API
Provides frontend components for monitoring dashboard and analytics
"""

# This file contains React component definitions for the enhanced API
# Copy these components to your React UI project

import json
from pathlib import Path

# Component 1: PerformanceDashboard Component
performance_dashboard_component = """
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
"""

# Component 2: AnalyticsChart Component
analytics_chart_component = """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AnalyticsChart = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/v1/analytics');
        setAnalyticsData(response.data.analytics);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch analytics:', err);
        setLoading(false);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading analytics...</div>;

  const gatewayData = Object.values(analyticsData)[0];

  return (
    <div className="analytics-chart">
      <h2>Analytics</h2>
      
      <div className="analytics-grid">
        <div className="analytics-item">
          <h4>Model</h4>
          <p>{gatewayData.model}</p>
        </div>
        
        <div className="analytics-item">
          <h4>Cache Hit Rate</h4>
          <p>{(gatewayData.cache_hit_rate * 100).toFixed(1)}%</p>
        </div>
        
        <div className="analytics-item">
          <h4>Cache Size</h4>
          <p>{gatewayData.cache_size} entries</p>
        </div>
        
        <div className="analytics-item">
          <h4>Total Requests</h4>
          <p>{gatewayData.total_requests}</p>
        </div>
        
        <div className="analytics-item">
          <h4>Avg Response Time</h4>
          <p>{gatewayData.avg_response_time.toFixed(2)}s</p>
        </div>
        
        <div className="analytics-item">
          <h4>Peak Response Time</h4>
          <p>{gatewayData.peak_response_time.toFixed(2)}s</p>
        </div>
      </div>

      <div className="query-distribution">
        <h3>Query Type Distribution</h3>
        <ul>
          {Object.entries(gatewayData.query_type_distribution).map(([type, count]) => (
            <li key={type}>{type}: {count}</li>
          ))}
        </ul>
      </div>

      <div className="cache-trend">
        <h3>Cache Hit Rate Trend</h3>
        <div className="trend-chart">
          {gatewayData.cache_hit_rate_trend.map((rate, index) => (
            <div key={index} className="trend-bar">
              <div 
                className="trend-fill" 
                style={{ width: `${rate * 100}%` }}
              />
              <span>{(rate * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AnalyticsChart;
"""

# Component 3: ConfigurationPanel Component
configuration_panel_component = """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ConfigurationPanel = () => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/v1/config');
        setConfig(response.data.configuration);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch config:', err);
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  if (loading) return <div>Loading configuration...</div>;

  return (
    <div className="configuration-panel">
      <h2>System Configuration</h2>
      
      <div className="config-section">
        <h3>Cache Configuration</h3>
        <p>Enabled: {config.cache.enabled ? 'Yes' : 'No'}</p>
        <p>Max Size: {config.cache.max_size} entries</p>
        <p>Semantic Threshold: {config.cache.semantic_threshold}</p>
      </div>

      <div className="config-section">
        <h3>Model Configuration</h3>
        <p>Default Model: {config.model.default_model}</p>
        <p>Temperature: {config.model.temperature}</p>
        <p>Max Tokens: {config.model.max_tokens}</p>
      </div>

      <div className="config-section">
        <h3>Performance Configuration</h3>
        <p>Performance Mode: {config.performance.performance_mode}</p>
        <p>Optimizations Enabled: {config.performance.enable_all_optimizations ? 'Yes' : 'No'}</p>
      </div>

      <div className="config-section">
        <h3>Monitoring Configuration</h3>
        <p>Analytics Enabled: {config.monitoring.analytics_enabled ? 'Yes' : 'No'}</p>
        <p>Performance Tracking: {config.monitoring.performance_tracking ? 'Yes' : 'No'}</p>
      </div>
    </div>
  );
};

export default ConfigurationPanel;
"""

# CSS Styles for the components
dashboard_styles = """
/* Performance Dashboard Styles */
.dashboard {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.metric-card {
  background: #f5f5f5;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.alerts-section {
  margin: 20px 0;
  padding: 15px;
  background: #fff3cd;
  border-radius: 8px;
}

.alert {
  padding: 10px;
  margin: 5px 0;
  border-radius: 4px;
}

.alert-warning {
  background: #ffc107;
  color: #000;
}

.alert-critical {
  background: #dc3545;
  color: #fff;
}

.configuration-section {
  margin: 20px 0;
  padding: 15px;
  background: #e9ecef;
  border-radius: 8px;
}

/* Analytics Chart Styles */
.analytics-chart {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.analytics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin: 20px 0;
}

.analytics-item {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  text-align: center;
}

.query-distribution ul {
  list-style: none;
  padding: 0;
}

.query-distribution li {
  padding: 5px 0;
  border-bottom: 1px solid #ddd;
}

.trend-chart {
  margin: 10px 0;
}

.trend-bar {
  height: 30px;
  background: #e9ecef;
  margin: 5px 0;
  border-radius: 4px;
  position: relative;
}

.trend-fill {
  height: 100%;
  background: #4CAF50;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.trend-bar span {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 12px;
  font-weight: bold;
}
"""

def save_ui_components():
    """Save UI components to files for integration."""
    components_dir = Path("ui_components")
    components_dir.mkdir(exist_ok=True)
    
    # Save React components
    with open(components_dir / "PerformanceDashboard.jsx", 'w') as f:
        f.write(performance_dashboard_component)
    
    with open(components_dir / "AnalyticsChart.jsx", 'w') as f:
        f.write(analytics_chart_component)
    
    with open(components_dir / "ConfigurationPanel.jsx", 'w') as f:
        f.write(configuration_panel_component)
    
    # Save CSS styles
    with open(components_dir / "dashboard.css", 'w') as f:
        f.write(dashboard_styles)
    
    # Save integration guide
    integration_guide = """
# UI Integration Guide

## Installation
1. Copy the React components from `ui_components/` to your React project
2. Copy the CSS styles to your project's stylesheet
3. Install axios: `npm install axios`

## Usage
```jsx
import PerformanceDashboard from './ui_components/PerformanceDashboard';
import AnalyticsChart from './ui_components/AnalyticsChart';
import ConfigurationPanel from './ui_components/ConfigurationPanel';

function App() {
  return (
    <div>
      <PerformanceDashboard />
      <AnalyticsChart />
      <ConfigurationPanel />
    </div>
  );
}
```

## API Endpoints Used
- GET /api/v1/dashboard - Performance dashboard data
- GET /api/v1/analytics - Analytics data
- GET /api/v1/config - Configuration data

## Features
- Real-time performance monitoring
- Automatic updates every 5-10 seconds
- Alert system for performance issues
- Configuration display
- Query type distribution
- Cache hit rate trends
"""
    
    with open(components_dir / "INTEGRATION_GUIDE.md", 'w') as f:
        f.write(integration_guide)
    
    print(f"UI components saved to {components_dir}")
    print("Integration guide available in ui_components/INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    save_ui_components()