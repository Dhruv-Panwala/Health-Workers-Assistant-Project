import React, { useState } from "react";
import ChartsPanel from "./ChartPanel";
import "./ChatInterface.css";

function ChatInterface() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [insightsAvailable, setInsightsAvailable] = useState(false);
  const [showCharts, setShowCharts] = useState(false);

  const API_URL = "https://health-workers-assistant-project.onrender.com/query";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    const userQuestion = input;
    setInput("");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          debug: false,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || "Something went wrong");
      }

      setResult({
        question: userQuestion,
        view: data.view || "records",
        columns: data.columns || [],
        rows: data.rows || [],
        row_count: data.row_count ?? (data.rows ? data.rows.length : 0),
      });

      setInsightsAvailable(true);
      setShowCharts(true);
    } catch (err) {
      setError(err.message || "Failed to fetch API");
      setInsightsAvailable(false);
      setShowCharts(false);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Helper: extract summary values safely
  const getSummaryValue = (label) => {
    if (!result || !result.columns || !result.rows || result.rows.length === 0) return null;
    const idx = result.columns.findIndex((c) => c.toLowerCase() === label.toLowerCase());
    if (idx === -1) return null;
    return result.rows[0]?.[idx];
  };

  const totalValue = result?.view === "summary" ? getSummaryValue("Total") : null;
  const recordsValue = result?.view === "summary" ? getSummaryValue("Records") : null;

  return (
    <div className="chat-container">
      <div className="chat-interface">
        {/* TOP: QUERY INPUT */}
        <form className="input-section input-top" onSubmit={handleSubmit}>
          <div className="input-container">
            <textarea
              className="input-box"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your health-related question here..."
              rows="3"
            />

            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || loading}
            >
              {loading ? "..." : (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z"
                    fill="currentColor"
                  />
                </svg>
              )}
            </button>
          </div>
        </form>

        {/* MAIN CONTENT AREA */}
        <div className={`content-grid ${showCharts && insightsAvailable ? "with-insights" : ""}`}>
          {/* OUTPUT */}
          <div className="output-section">
            <div className="output-box">
              {/* Error */}
              {error && <div className="error-text">❌ {error}</div>}

              {/* Loading */}
              {loading && <div className="loading-text">Fetching results...</div>}

              {/* Result */}
              {!loading && result && (
                <>
                  {/* SUMMARY VIEW */}
                  {result.view === "summary" ? (
                    <div className="summary-card">
                      <div className="summary-item">
                        <div className="summary-label">Total</div>
                        <div className="summary-value">
                          {totalValue === null ? "-" : totalValue.toFixed(3)}
                        </div>
                      </div>

                      <div className="summary-item">
                        <div className="summary-label">Records</div>
                        <div className="summary-value">
                          {recordsValue === null ? "-" : recordsValue}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* RECORDS TABLE VIEW */}
                      {result.columns.length > 0 ? (
                        <div className="table-wrapper">
                          <table className="result-table">
                            <thead>
                              <tr>
                                {result.columns.map((col, idx) => (
                                  <th key={idx}>{col}</th>
                                ))}
                              </tr>
                            </thead>

                            <tbody>
                              {result.rows.length > 0 ? (
                                result.rows.map((row, rIdx) => (
                                  <tr key={rIdx}>
                                    {result.columns.map((_, cIdx) => (
                                      <td key={cIdx}>
                                        {row?.[cIdx] === null || row?.[cIdx] === undefined
                                          ? "-"
                                          : String(row[cIdx])}
                                      </td>
                                    ))}
                                  </tr>
                                ))
                              ) : (
                                <tr>
                                  <td colSpan={result.columns.length} style={{ padding: "14px" }}>
                                    No rows returned yet (but columns are available).
                                  </td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="empty-text">No columns returned from API.</div>
                      )}
                    </>
                  )}
                </>
              )}

              {/* Default */}
              {!loading && !error && !result && (
                <pre className="output-text">Ask a question to see results here.</pre>
              )}
            </div>
          </div>

          {/* INSIGHTS */}
          {showCharts && insightsAvailable && (
            <div className="insights-section">
              <ChartsPanel />
            </div>
          )}
        </div>

        {/* INSIGHTS BUTTON */}
        <div className="insights-float">
          <span
            className="tooltip-wrapper"
            data-tooltip={
              insightsAvailable
                ? showCharts
                  ? "Hide insights"
                  : "Show insights"
                : "Insights not available for this query yet"
            }
          >
            <button
              className="insights-btn"
              disabled={!insightsAvailable}
              onClick={() => setShowCharts((prev) => !prev)}
            >
              {showCharts ? "Hide Insights" : "Insights"}
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
