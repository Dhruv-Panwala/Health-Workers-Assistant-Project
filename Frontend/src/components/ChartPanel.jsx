import React, { useMemo } from "react";
import "./ChartPanel.css";

function ChartsPanel({ result }) {
  const { view, columns = [], rows = [] } = result || {};

  const colIndex = useMemo(() => {
    const map = {};
    columns.forEach((c, i) => (map[c] = i));
    return map;
  }, [columns]);

  const hasOrg = "Organisation Unit" in colIndex;
  const hasMetric = "Metric" in colIndex;
  const hasValue = "Value" in colIndex;

  const parsedRows = useMemo(() => {
    if (!rows || !rows.length || !hasValue) return [];

    return rows.map((r) => {
      const valueRaw = r[colIndex["Value"]];
      const valueNum =
        valueRaw === null || valueRaw === undefined ? 0 : Number(valueRaw);

      return {
        org: hasOrg ? r[colIndex["Organisation Unit"]] : null,
        metric: hasMetric ? r[colIndex["Metric"]] : null,
        value: isNaN(valueNum) ? 0 : valueNum,
      };
    });
  }, [rows, colIndex, hasOrg, hasMetric, hasValue]);

  const uniqueOrgCount = useMemo(() => {
    if (!hasOrg) return 0;
    return new Set(parsedRows.map((x) => x.org)).size;
  }, [parsedRows, hasOrg]);

  const uniqueMetricCount = useMemo(() => {
    if (!hasMetric) return 0;
    return new Set(parsedRows.map((x) => x.metric)).size;
  }, [parsedRows, hasMetric]);

  const chartMode = useMemo(() => {
    if (!hasValue) return "none";
    if (uniqueOrgCount > 1) return "top_orgs";
    if (uniqueMetricCount > 1) return "top_metrics";
    return "none";
  }, [hasValue, uniqueOrgCount, uniqueMetricCount]);

  const top10Data = useMemo(() => {
    if (chartMode === "none") return [];

    const groupKey = chartMode === "top_orgs" ? "org" : "metric";
    const map = new Map();

    parsedRows.forEach((row) => {
      const key = row[groupKey] || "Unknown";
      map.set(key, (map.get(key) || 0) + row.value);
    });

    return Array.from(map.entries())
      .map(([name, total]) => ({ name, total }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 10);
  }, [parsedRows, chartMode]);

  // Summary view -> disable insights
  if (!result || view === "summary") {
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

  if (!rows.length) {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">No rows returned, so no charts yet.</div>
        </div>
      </div>
    );
  }

  if (chartMode === "none") {
    return (
      <div className="charts-panel">
        <div className="chart-card">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">
            Not enough variation in the results to build a chart.
          </div>
        </div>
      </div>
    );
  }

  const title =
    chartMode === "top_orgs"
      ? "Top 10 Organisations (by total value)"
      : "Top 10 Metrics (by total value)";

  const max = top10Data[0]?.total || 1;

  return (
    <div className="charts-panel">
      <div className="chart-card chart-card-big">
        <div className="chart-title">{title}</div>

        <div className="chart-scroll">
          {top10Data.map((item, idx) => {
            const pct = (item.total / max) * 100;

            return (
              <div className="bar-row" key={idx}>
                <div className="bar-row-top">
                  <span className="bar-label">{item.name}</span>
                  <span className="bar-value">{Math.round(item.total)}</span>
                </div>

                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${pct}%` }} />
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
