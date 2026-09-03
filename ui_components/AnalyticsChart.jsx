
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
