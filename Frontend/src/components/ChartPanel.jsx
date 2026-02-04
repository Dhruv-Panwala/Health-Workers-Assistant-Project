import React from "react";
import "./ChartPanel.css";

function ChartsPanel({ insights, view }) {
  // 1️⃣ Summary queries never have insights
  if (view === "summary") {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">
            Insights are disabled for summary queries.
          </div>
        </div>
      </div>
    );
  }

  if (!insights || !insights.data) {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">Loading insights…</div>
        </div>
      </div>
    );
  }


  // 3️⃣ Validate insights structure
  const mode = insights.mode;
  const data = Array.isArray(insights.data) ? insights.data : [];

  const isInvalid =
    mode === "none" || !mode;

  if (isInvalid) {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">
            No insights available for the current query.
          </div>
        </div>
      </div>
    );
  }

  // 4️⃣ Determine title
  const title =
    mode === "top_orgs"
      ? "Top Organisations (by total value)"
      : mode === "top_metrics"
      ? "Top Metrics (by total value)"
      : "Insights";

  // 5️⃣ Compute scale
  const maxValue = Math.max(...data.map((d) => d.total || 0)) || 1;

  return (
    <div className="charts-panel">
      <div className="chart-card chart-card-big">
        <div className="chart-title">{title}</div>

        <div className="chart-scroll">
          {data.map((item, idx) => {
            const total = item.total || 0;
            const pct = Math.max(2, (total / maxValue) * 100); // never show 0-width bars

            return (
              <div className="bar-row" key={idx}>
                <div className="bar-row-top">
                  <span className="bar-label">{item.name}</span>
                  <span className="bar-value">{Math.round(total)}</span>
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
