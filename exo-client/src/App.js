import React, { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError("");

    try {
      // Connect to your FastAPI Backend
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Process data for the Graph (Map array to Objects)
      const graphData = response.data.flux_data.map((value, index) => ({
        time: index,
        flux: value
      }));

      setResult({
        prediction: response.data.prediction,
        confidence: (response.data.confidence * 100).toFixed(2),
        chartData: graphData
      });

    } catch (err) {
      console.error(err);
      setError("Failed to connect to the server. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Exoplanet Finder</h1>
      </header>

      <div className="control-panel">
        <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />
        <button onClick={handleUpload} disabled={loading} className="analyze-btn">
          {loading ? "Analyzing..." : "Analyze Light Curve"}
        </button>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="results-section">
          <div className={`status-card ${result.prediction.includes("No") ? "negative" : "positive"}`}>
            <h2>{result.prediction}</h2>
            <p>Confidence: <strong>{result.confidence}%</strong></p>
          </div>

          <div className="chart-container">
            <h3>Light Curve Analysis</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={result.chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="time" stroke="#7f8c8d" />
                <YAxis stroke="#7f8c8d" domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e0e0e0' }}
                  itemStyle={{ color: '#2c3e50' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="flux" 
                  stroke="#3498db" 
                  strokeWidth={2} 
                  dot={false} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;