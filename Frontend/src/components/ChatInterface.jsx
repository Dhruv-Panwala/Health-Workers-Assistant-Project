import React, { useState, useEffect } from "react";
import ChartsPanel from "./ChartPanel";
import "./ChatInterface.css";

function ChatInterface() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [insightsAvailable, setInsightsAvailable] = useState(false);
  const [showCharts, setShowCharts] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(200);

  const API_URL = "https://health-workers-assistant-project.onrender.com/query";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuestion = input.trim();
    setCurrentQuery(userQuestion);
    setLoading(true);
    setError("");
    setResult(null);
    setInput("");
    setCurrentPage(1);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          debug: false,
          page: 1,
          page_size: rowsPerPage
    }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || "Something went wrong");
      }

      const view = data.view || "records";
      setResult({
        question: userQuestion,
        view,
        columns: data.columns || [],
        rows: data.rows || [],
        row_count: data.row_count ?? (data.rows ? data.rows.length : 0),
      });

      const isSummary = view === "summary";
      setInsightsAvailable(!isSummary);
      setShowCharts(!isSummary);
    } catch (err) {
      setError(err.message || "Failed to fetch API");
      setInsightsAvailable(false);
      setShowCharts(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
  if (!currentQuery || currentPage === 1) return;

  const fetchPage = async () => {
    setLoading(true);
    setError("");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: currentQuery,
          debug: false,
          page: currentPage,
          page_size: rowsPerPage,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || "Something went wrong");
      }

      setResult(prev => ({
        ...prev,
        rows: data.rows || [],
        row_count: data.total_count || data.row_count || (data.rows ? data.rows.length : 0),
      }));
    } catch (err) {
      setError(err.message || "Failed to fetch API");
    } finally {
      setLoading(false);
    }
  };

  fetchPage();
}, [currentPage]);


  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const getSummaryValue = (label) => {
    if (!result || !result.columns || !result.rows || result.rows.length === 0) return null;
    const idx = result.columns.findIndex(c => String(c).toLowerCase() === label.toLowerCase());
    if (idx === -1) return null;
    return result.rows[0]?.[idx];
  };

  const totalValue = result?.view === "summary" ? getSummaryValue("Total") : null;
  const recordsValue = result?.view === "summary" ? getSummaryValue("Records") : null;

  const formatTotal = (val) => {
    if (val === null || val === undefined) return "-";
    const num = Number(val);
    if (Number.isNaN(num)) return String(val);
    return Math.abs(num - Math.round(num)) < 0.000001 ? String(Math.round(num)) : num.toFixed(2);
  };

  const paginatedRows = result?.rows.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);
  const totalPages = Math.ceil((result?.row_count || 0) / rowsPerPage);

  const canGoPrev = currentPage > 1;
  const canGoNext = currentPage < totalPages;

  return (
    <div className="chat-container">
      <div className="chat-interface">
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
            <button type="submit" className="send-button" disabled={!input.trim() || loading}>
              {loading ? "..." : <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="currentColor"/></svg>}
            </button>
          </div>
        </form>

        <div className={`content-grid ${showCharts && insightsAvailable ? "with-insights" : ""}`}>
          <div className="output-section">
            {currentQuery && <div className="query-display"><span className="query-label">Query:</span><div className="query-text">{currentQuery}</div></div>}
            <div className="output-box">
              {error && <div className="error-text">Error: Please try another query</div>}
              {loading && <div className="loading-text">Fetching results...</div>}

              {!loading && result && (
                result.view === "summary" ? (
                  <div className="summary-card">
                    <div className="summary-title">Summary Results</div>
                    <div className="summary-grid">
                      <div className="summary-item">
                        <div className="summary-label">Total recorded</div>
                        <div className="summary-value">{formatTotal(totalValue)}</div>
                      </div>
                      <div className="summary-item">
                        <div className="summary-label">Records found</div>
                        <div className="summary-value">{recordsValue === null ? "-" : recordsValue}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  result.columns.length > 0 ? (
                    <>
                      <div className="table-wrapper">
                        <table className="result-table">
                          <thead>
                            <tr>{result.columns.map((col, idx) => <th key={idx}>{col}</th>)}</tr>
                          </thead>
                          <tbody>
                            {paginatedRows.length > 0 ? paginatedRows.map((row, rIdx) => (
                              <tr key={rIdx}>{result.columns.map((_, cIdx) => <td key={cIdx}>{row?.[cIdx] ?? "-"}</td>)}</tr>
                            )) : (
                              <tr><td colSpan={result.columns.length} style={{ padding: "14px" }}>No rows returned yet.</td></tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </>
                  ) : (
                    <div className="empty-text">No columns returned from API.</div>
                  )
                )
              )}

              {!loading && !error && !result && (
                <pre className="output-text">Ask a question to see results here.</pre>
              )}
            </div>
            <div className="pagination-controls">
                <button onClick={() => setCurrentPage(p => p - 1)} disabled={!canGoPrev}>Previous</button>
                <span>Page {currentPage} of {totalPages}</span>
                <button onClick={() => setCurrentPage(p => p + 1)} disabled={!canGoNext}>Next</button>
             </div> 
          </div>

          {showCharts && insightsAvailable && (
            <div className="insights-section">
              <ChartsPanel result={result} currentQuery={currentQuery} noLimitMode={true} />
            </div>
          )}
        </div>

        <div className="insights-float">
          <span className="tooltip-wrapper" data-tooltip={insightsAvailable ? (showCharts ? "Hide insights" : "Show insights") : "Insights not available for this query yet"}>
            <button className="insights-btn" disabled={!insightsAvailable} onClick={() => setShowCharts(p => !p)}>
              {showCharts ? "Hide Insights" : "Insights"}
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
