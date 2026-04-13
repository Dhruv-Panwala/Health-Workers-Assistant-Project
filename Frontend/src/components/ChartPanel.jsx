import React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./ChartPanel.css";

const CHART_COLORS = [
  "#0f766e",
  "#2563eb",
  "#dc2626",
  "#ea580c",
  "#7c3aed",
  "#16a34a",
  "#0891b2",
  "#be123c",
];

const AXIS_NUMBER_FORMATTER = new Intl.NumberFormat("en-GB", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function toNumber(value, fallback = 0) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value.replace(/,/g, "").trim());
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

function formatValue(value) {
  return Math.round(toNumber(value, 0)).toLocaleString();
}

function formatAxisValue(value) {
  return AXIS_NUMBER_FORMATTER.format(toNumber(value, 0));
}

function prettifyLabel(label) {
  return String(label || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function truncateLabel(label, maxLength = 22) {
  const text = String(label || "");
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function formatDateLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) {
    return "";
  }

  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }

  return date.toLocaleString("en-GB", {
    month: "short",
    year: "numeric",
  });
}

function normalizeDashboardInsights(rawInsights) {
  if (!rawInsights || typeof rawInsights !== "object") {
    return null;
  }

  if (rawInsights.mode === "dashboard") {
    return {
      mode: "dashboard",
      cards: {
        total: toNumber(rawInsights?.cards?.total, 0),
        metrics: toNumber(rawInsights?.cards?.metrics, 0),
        orgs: toNumber(rawInsights?.cards?.orgs, 0),
      },
      charts: Array.isArray(rawInsights?.charts)
        ? rawInsights.charts.map((chart) => {
            if (chart?.type === "line_trend") {
              return {
                ...chart,
                data: Array.isArray(chart.data)
                  ? chart.data.map((point) => ({
                      date: String(point?.date || ""),
                      total: toNumber(point?.total, 0),
                    }))
                  : [],
              };
            }

            if (chart?.type === "bar_metrics" || chart?.type === "bar_orgs") {
              return {
                ...chart,
                data: Array.isArray(chart.data)
                  ? chart.data.map((point) => ({
                      name: String(point?.name || ""),
                      total: toNumber(point?.total, 0),
                    }))
                  : [],
              };
            }

            if (chart?.type === "multi_line_metric_trend") {
              return {
                ...chart,
                series: Array.isArray(chart.series)
                  ? chart.series.map((seriesEntry) => ({
                      metric: String(seriesEntry?.metric || ""),
                      data: Array.isArray(seriesEntry?.data)
                        ? seriesEntry.data.map((point) => ({
                            date: String(point?.date || ""),
                            total: toNumber(point?.total, 0),
                          }))
                        : [],
                    }))
                  : [],
              };
            }

            return chart;
          })
        : [],
    };
  }

  if (Array.isArray(rawInsights.preview) && rawInsights.preview.length > 0) {
    return {
      mode: "preview",
      preview: rawInsights.preview,
      matchedRows: toNumber(rawInsights.matched_rows, rawInsights.preview.length),
      filters: rawInsights.filters || {},
      originalMode: rawInsights.mode || "records",
    };
  }

  if (rawInsights.mode === "none") {
    return { mode: "none" };
  }

  return {
    mode: "unsupported",
    originalMode: rawInsights.mode || "unknown",
  };
}

function previewValueCount(rows, key, predicate) {
  return rows.reduce((count, row) => count + (predicate(row?.[key]) ? 1 : 0), 0);
}

function isDateLike(value) {
  if (value === null || value === undefined || value === "") {
    return false;
  }

  const date = new Date(String(value));
  return !Number.isNaN(date.getTime());
}

function buildPreviewChart(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return null;
  }

  const keys = Object.keys(rows[0] || {});
  if (!keys.length) {
    return null;
  }

  const numericKey = [...keys]
    .map((key) => ({
      key,
      count: previewValueCount(rows, key, (value) => Number.isFinite(toNumber(value, Number.NaN))),
      score:
        previewValueCount(rows, key, (value) => Number.isFinite(toNumber(value, Number.NaN))) +
        (/(total|value|count|case|sum|latest|points)/i.test(key) ? 3 : 0) -
        (/(id|uid|code)/i.test(key) ? 3 : 0),
    }))
    .filter((entry) => entry.count > 0)
    .sort((left, right) => right.score - left.score)[0]?.key;

  if (!numericKey) {
    return null;
  }

  const dateKey = keys
    .filter((key) => key !== numericKey)
    .map((key) => ({
      key,
      count: previewValueCount(rows, key, isDateLike),
      score:
        previewValueCount(rows, key, isDateLike) +
        (/(period|date|month|year|start|end)/i.test(key) ? 2 : 0),
    }))
    .filter((entry) => entry.count > 1)
    .sort((left, right) => right.score - left.score)[0]?.key;

  const labelKey = keys
    .filter((key) => key !== numericKey && key !== dateKey)
    .map((key) => ({
      key,
      count: previewValueCount(
        rows,
        key,
        (value) => typeof value === "string" && value.trim().length > 0
      ),
      score:
        previewValueCount(
          rows,
          key,
          (value) => typeof value === "string" && value.trim().length > 0
        ) +
        (/(name|metric|org|indicator|facility|unit)/i.test(key) ? 3 : 0) -
        (/(id|uid|code)/i.test(key) ? 3 : 0),
    }))
    .filter((entry) => entry.count > 0)
    .sort((left, right) => right.score - left.score)[0]?.key;

  if (dateKey) {
    return {
      type: "line_preview",
      title: `${prettifyLabel(numericKey)} Preview`,
      data: rows
        .map((row) => ({
          label: formatDateLabel(row?.[dateKey]),
          sortKey: String(row?.[dateKey] || ""),
          value: toNumber(row?.[numericKey], 0),
        }))
        .filter((row) => row.label)
        .sort((left, right) => left.sortKey.localeCompare(right.sortKey))
        .slice(0, 12),
    };
  }

  if (labelKey) {
    return {
      type: "bar_preview",
      title: `${prettifyLabel(numericKey)} by ${prettifyLabel(labelKey)}`,
      data: rows
        .map((row) => ({
          label: String(row?.[labelKey] || ""),
          value: toNumber(row?.[numericKey], 0),
        }))
        .filter((row) => row.label)
        .sort((left, right) => right.value - left.value)
        .slice(0, 6),
    };
  }

  return null;
}

function renderTooltipValue(value) {
  return [formatValue(value), "Total"];
}

function renderCards(cards) {
  if (!cards) {
    return null;
  }

  const cardItems = [
    { label: "Total", value: formatValue(cards.total), accent: "teal" },
    { label: "Metrics", value: formatValue(cards.metrics), accent: "blue" },
    { label: "Organisations", value: formatValue(cards.orgs), accent: "amber" },
  ];

  return (
    <div className="kpi-grid">
      {cardItems.map((item) => (
        <div
          key={item.label}
          className={`kpi-card kpi-card-${item.accent}`}
        >
          <div className="kpi-label">{item.label}</div>
          <div className="kpi-value">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

function renderBreakdownChart(chart) {
  const data = Array.isArray(chart.data)
    ? chart.data
        .map((item) => ({
          name: String(item?.name || ""),
          shortName: truncateLabel(item?.name || ""),
          total: toNumber(item?.total, 0),
        }))
        .filter((item) => item.name)
        .sort((left, right) => right.total - left.total)
    : [];

  const chartHeight = Math.max(280, data.length * 52);

  return (
    <div className="chart-shell" style={{ height: chartHeight }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 6, right: 16, left: 8, bottom: 6 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tickFormatter={formatAxisValue} />
          <YAxis
            type="category"
            dataKey="shortName"
            width={120}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={renderTooltipValue}
            labelFormatter={(_label, payload) => payload?.[0]?.payload?.name || ""}
          />
          <Bar dataKey="total" radius={[0, 10, 10, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`${entry.name}-${index}`}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function renderTrendChart(chart) {
  const data = Array.isArray(chart.data)
    ? chart.data.map((item) => ({
        date: String(item?.date || ""),
        label: formatDateLabel(item?.date),
        total: toNumber(item?.total, 0),
      }))
    : [];

  return (
    <div className="chart-shell chart-shell-tall">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <defs>
            <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0f766e" stopOpacity={0.28} />
              <stop offset="95%" stopColor="#0f766e" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis tickFormatter={formatAxisValue} tickLine={false} axisLine={false} />
          <Tooltip formatter={renderTooltipValue} />
          <Area
            type="monotone"
            dataKey="total"
            stroke="#0f766e"
            fill="url(#trendFill)"
            strokeWidth={3}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function renderMultiTrendChart(chart) {
  const series = Array.isArray(chart.series) ? chart.series : [];
  const merged = {};

  series.forEach((seriesEntry) => {
    (Array.isArray(seriesEntry.data) ? seriesEntry.data : []).forEach((point) => {
      const key = String(point?.date || "");
      if (!key) {
        return;
      }

      if (!merged[key]) {
        merged[key] = {
          date: key,
          label: formatDateLabel(key),
        };
      }

      merged[key][seriesEntry.metric] = toNumber(point?.total, 0);
    });
  });

  const data = Object.values(merged).sort((left, right) =>
    String(left.date).localeCompare(String(right.date))
  );

  return (
    <div className="chart-shell chart-shell-tall">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis tickFormatter={formatAxisValue} tickLine={false} axisLine={false} />
          <Tooltip formatter={(value) => formatValue(value)} />
          <Legend />
          {series.map((seriesEntry, index) => (
            <Line
              key={`${seriesEntry.metric}-${index}`}
              type="monotone"
              dataKey={seriesEntry.metric}
              stroke={CHART_COLORS[index % CHART_COLORS.length]}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
              name={seriesEntry.metric}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function renderPreviewChart(previewChart) {
  if (!previewChart || !Array.isArray(previewChart.data) || !previewChart.data.length) {
    return null;
  }

  if (previewChart.type === "line_preview") {
    return (
      <div className="chart-shell">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={previewChart.data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <defs>
              <linearGradient id="previewTrendFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#2563eb" stopOpacity={0.24} />
                <stop offset="95%" stopColor="#2563eb" stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={20} />
            <YAxis tickFormatter={formatAxisValue} tickLine={false} axisLine={false} />
            <Tooltip formatter={renderTooltipValue} />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#2563eb"
              fill="url(#previewTrendFill)"
              strokeWidth={3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="chart-shell">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={previewChart.data} margin={{ top: 8, right: 16, left: 0, bottom: 18 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tickFormatter={(value) => truncateLabel(value, 14)}
            tickLine={false}
            axisLine={false}
            interval={0}
            angle={-18}
            textAnchor="end"
            height={60}
          />
          <YAxis tickFormatter={formatAxisValue} tickLine={false} axisLine={false} />
          <Tooltip formatter={renderTooltipValue} />
          <Bar dataKey="value" radius={[10, 10, 0, 0]} fill="#2563eb" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function renderPreviewFilters(filters) {
  const chips = Object.entries(filters || {})
    .flatMap(([key, value]) => {
      if (value === null || value === undefined || value === "") {
        return [];
      }
      if (Array.isArray(value) && !value.length) {
        return [];
      }

      const displayValue = Array.isArray(value) ? value.join(", ") : String(value);
      if (!displayValue.trim()) {
        return [];
      }

      return `${prettifyLabel(key)}: ${displayValue}`;
    })
    .slice(0, 6);

  if (!chips.length) {
    return null;
  }

  return (
    <div className="preview-filter-row">
      {chips.map((chip) => (
        <span className="preview-filter-chip" key={chip}>
          {chip}
        </span>
      ))}
    </div>
  );
}

function renderPreviewInsights(insights) {
  const previewChart = buildPreviewChart(insights.preview);

  return (
    <div className="charts-panel">
      <div className="chart-card chart-card-big chart-card-compact">
        <div className="chart-title-row">
          <div className="chart-title">Insights Preview</div>
          <div className="chart-subtitle">
            Sample of {formatValue(insights.matchedRows)} matched rows
          </div>
        </div>

        {renderPreviewFilters(insights.filters)}

        {previewChart && (
          <div className="preview-chart-panel">
            <div className="preview-chart-title">{previewChart.title}</div>
            {renderPreviewChart(previewChart)}
          </div>
        )}

        <div className="preview-grid">
          {insights.preview.slice(0, 4).map((item, index) => (
            <div className="preview-card" key={`preview-${index}`}>
              {Object.entries(item).map(([label, value]) => (
                <div className="preview-row" key={label}>
                  <span className="preview-label">{prettifyLabel(label)}</span>
                  <span className="preview-value">{String(value)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function renderDashboardInsights(insights, view) {
  const charts = Array.isArray(insights.charts) ? insights.charts : [];

  return (
    <div className="charts-panel">
      {renderCards(insights.cards)}

      {charts.map((chart, index) => (
        <div className="chart-card chart-card-big" key={`${chart.type}-${index}`}>
          <div className="chart-title-row">
            <div className="chart-title">{chart.title}</div>
            <div className="chart-subtitle">
              {view === "summary"
                ? "Summary insight"
                : view === "explainable"
                  ? "Explainable insight"
                  : "Offline analytics"}
            </div>
          </div>

          {chart.type === "line_trend" && renderTrendChart(chart)}
          {(chart.type === "bar_metrics" || chart.type === "bar_orgs") &&
            renderBreakdownChart(chart)}
          {chart.type === "multi_line_metric_trend" &&
            renderMultiTrendChart(chart)}
        </div>
      ))}

      {!charts.length && (
        <div className="chart-card chart-card-compact">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">No chartable insight points available.</div>
        </div>
      )}
    </div>
  );
}

function ChartsPanel({ insights, view }) {
  const normalizedInsights = normalizeDashboardInsights(insights);

  if (!normalizedInsights || normalizedInsights.mode === "none") {
    return (
      <div className="charts-panel">
        <div className="chart-card chart-card-compact">
          <div className="chart-title">Insights</div>
          <div className="chart-empty">No insights available.</div>
        </div>
      </div>
    );
  }

  if (normalizedInsights.mode === "dashboard") {
    return renderDashboardInsights(normalizedInsights, view);
  }

  if (normalizedInsights.mode === "preview") {
    return renderPreviewInsights(normalizedInsights);
  }

  return (
    <div className="charts-panel">
      <div className="chart-card chart-card-compact">
        <div className="chart-title">Insights</div>
        <div className="chart-empty">
          Unsupported insight mode: {normalizedInsights.originalMode}
        </div>
      </div>
    </div>
  );
}

export default ChartsPanel;
