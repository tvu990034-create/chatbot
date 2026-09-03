
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
