import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";
import "./ChartPanel.css";

function ChartsPanel({ insights, view }) {

  // -------------------------------------------------
  // Helpers
  // -------------------------------------------------

  const formatMonth = (isoDate) => {
    const d = new Date(isoDate);
    return d.toLocaleString("en-GB", {
      month: "short",
      year: "numeric",
    });
  };

  // ⭐ Fill missing months with 0 values (frontend fix)
  const fillMissingMonths = (data) => {
  if (!data || data.length === 0) return [];

  const valueMap = new Map(
    data.map(d => [d.date.slice(0, 7), Math.floor(d.total || 0)])
  );

  const start = new Date(data[0].date);
  const end = new Date(data[data.length - 1].date);

  // force to first day of month
  let y = start.getFullYear();
  let m = start.getMonth();

  const endY = end.getFullYear();
  const endM = end.getMonth();

  const result = [];

  while (y < endY || (y === endY && m <= endM)) {
    const key = `${y}-${String(m + 1).padStart(2, "0")}`;

    result.push({
      date: `${key}-01`,   // safe ISO-like string
      total: valueMap.get(key) || 0
    });

    // increment month safely
    m += 1;
    if (m > 11) {
      m = 0;
      y += 1;
    }
  }

  return result;
  };


  // -------------------------------------------------
  // Empty states
  // -------------------------------------------------

  if (!insights) {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">No insights available.</div>
        </div>
      </div>
    );
  }

  const mode = insights.mode;
  const rawData = Array.isArray(insights.data) ? insights.data : [];

  if (!mode || mode === "none" || rawData.length === 0) {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">
            No insights available for this query.
          </div>
        </div>
      </div>
    );
  }

  // -------------------------------------------------
  // Titles
  // -------------------------------------------------

  const chartTitles = {
    top_orgs: "Top Organisations (by total value)",
    top_metrics: "Top Metrics (by total value)",
    single_total: "Total Summary Value",
    trend: "Monthly Trend"
  };

  const title = chartTitles[mode] || "Insights";

  // =====================================================
  // ⭐ TREND → LINE CHART
  // =====================================================

  if (mode === "trend") {

    const filled = fillMissingMonths(rawData);

    const formatted = filled.map(d => ({
      ...d,
      dateLabel: formatMonth(d.date),
      total: Math.floor(d.total || 0)
    }));

    return (
      <div className="charts-panel">
        <div className="chart-card chart-card-big">

          <div className="chart-title">
            {title}
            {view === "summary" && (
              <span className="chart-subtitle">(Summary Insights)</span>
            )}
          </div>

          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={formatted}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dateLabel" />
              <YAxis allowDecimals={false} />
              <Tooltip
                formatter={(v) => v.toLocaleString()}
              />
              <Line
                type="monotone"
                dataKey="total"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>

        </div>
      </div>
    );
  }

  // =====================================================
  // ⭐ BARS → categorical insights
  // =====================================================

  const data = rawData.map(d => ({
    ...d,
    total: Math.floor(d.total || 0)
  }));

  const maxValue = Math.max(...data.map(d => d.total)) || 1;

  return (
    <div className="charts-panel">
      <div className="chart-card chart-card-big">

        <div className="chart-title">
          {title}
          {view === "summary" && (
            <span className="chart-subtitle">(Summary Insights)</span>
          )}
        </div>

        <div className="chart-scroll">
          {data.map((item, idx) => {
            const pct = Math.max(2, (item.total / maxValue) * 100);

            return (
              <div className="bar-row" key={idx}>
                <div className="bar-row-top">
                  <span className="bar-label">{item.name}</span>
                  <span className="bar-value">
                    {item.total.toLocaleString()}
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

      </div>
    </div>
  );
}

export default ChartsPanel;
