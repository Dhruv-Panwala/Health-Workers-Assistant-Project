import React from "react";
import "./ChartPanel.css";

function ChartsPanel() {
  return (
    <div className="charts-panel">
      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-title">Top Symptoms Mentioned</div>
          <div className="chart">
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-title">Confidence</div>
          <div className="chart">
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChartsPanel;