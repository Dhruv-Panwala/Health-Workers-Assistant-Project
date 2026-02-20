import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from "recharts";

import "./ChartPanel.css";

function ChartsPanel({ insights, view }) {
  const LINE_COLORS = [
  "#2563eb", 
  "#dc2626", 
  "#16a34a", 
  "#7c3aed", 
  "#ea580c", 
  "#0f766e", 
  "#9333ea", 
  "#be123c" 
  ];
  if (!insights || insights.mode === "none") {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">No insights available.</div>
        </div>
      </div>
    );
  }

  // -----------------------------
  // Helpers
  // -----------------------------
  const formatMonth = (isoDate) => {
    const d = new Date(isoDate);
    return d.toLocaleString("en-GB", {
      month: "short",
      year: "numeric"
    });
  };

  // -----------------------------
  // Render KPI Cards
  // -----------------------------
  const renderCards = (cards) => {
    if (!cards) return null;

    return (
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Total</div>
          <div className="kpi-value">
            {Math.floor(cards.total).toLocaleString()}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Metrics</div>
          <div className="kpi-value">{cards.metrics}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Organisations</div>
          <div className="kpi-value">{cards.orgs}</div>
        </div>
      </div>
    );
  };

  // -----------------------------
  // Render Bar Chart List
  // -----------------------------
  const renderBars = (chart) => {
    const data = chart.data || [];
    const maxValue = Math.max(...data.map((d) => d.total)) || 1;

    return (
      <div className="chart-scroll">
        {data.map((item, idx) => {
          const pct = Math.max(2, (item.total / maxValue) * 100);

          return (
            <div className="bar-row" key={idx}>
              <div className="bar-row-top">
                <span className="bar-label">{item.name}</span>
                <span className="bar-value">
                  {Math.floor(item.total).toLocaleString()}
                </span>
              </div>

              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // -----------------------------
  // Render Single Line Trend
  // -----------------------------
  const renderTrend = (chart) => {
    const formatted = (chart.data || []).map((d) => ({
      ...d,
      dateLabel: formatMonth(d.date),
      total: Math.floor(d.total || 0)
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="dateLabel" />
          <YAxis allowDecimals={false} />
          <Tooltip formatter={(v) => v.toLocaleString()} />
          <Line type="monotone" dataKey="total" strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  // -----------------------------
  // Render Multi-Series Trend
  // -----------------------------
  const renderMultiTrend = (chart) => {
    const series = chart.series || [];

    // Flatten all dates
    const merged = {};
    series.forEach((s) => {
      s.data.forEach((pt) => {
        if (!merged[pt.date]) merged[pt.date] = { date: pt.date };
        merged[pt.date][s.metric] = pt.total;
      });
    });

    const finalData = Object.values(merged)
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .map((d) => ({
        ...d,
        dateLabel: formatMonth(d.date)
      }));

    return (
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={finalData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="dateLabel" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Legend/>

          {series.map((s, idx) => (
            <Line
              key={idx}
              type="monotone"
              dataKey={s.metric}
              stroke={LINE_COLORS[idx % LINE_COLORS.length]}
              strokeWidth={2}
              dot={false}
              name={s.metric}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  };

  // ======================================================
  // DASHBOARD MODE (NEW)
  // ======================================================
  if (insights.mode === "dashboard") {
    return (
      <div className="charts-panel">
        {/* KPI Cards */}
        {renderCards(insights.cards)}

        {/* Charts */}
        {insights.charts.map((chart, idx) => (
          <div className="chart-card chart-card-big" key={idx}>
            <div className="chart-title">
              {chart.title}
              {view === "summary" && (
                <span className="chart-subtitle">(Summary Insights)</span>
              )}
            </div>

            {/* Chart Types */}
            {chart.type === "line_trend" && renderTrend(chart)}
            {chart.type === "bar_metrics" && renderBars(chart)}
            {chart.type === "bar_orgs" && renderBars(chart)}
            {chart.type === "multi_line_metric_trend" &&
              renderMultiTrend(chart)}
          </div>
        ))}
      </div>
    );
  }

  // ======================================================
  // BACKWARD COMPATIBILITY (OLD MODES)
  // ======================================================
  return (
    <div className="charts-panel">
      <div className="chart-card">
        <div className="chart-title">Insights</div>
        <div className="chart-empty">
          Unsupported insight mode: {insights.mode}
        </div>
      </div>
    </div>
  );
}

export default ChartsPanel;
