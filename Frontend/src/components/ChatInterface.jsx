import React, { useState, useEffect, useRef } from "react";
import ChartsPanel from "./ChartPanel";
import "./ChatInterface.css";

function ChatInterface() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);

  const [loadingTable, setLoadingTable] = useState(false);
  const [loadingInsights, setLoadingInsights] = useState(false);

  const [error, setError] = useState("");
  const [insightsAvailable, setInsightsAvailable] = useState(false);
  const [showCharts, setShowCharts] = useState(false);

  const [currentQuery, setCurrentQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(100);

  const pageAbortRef = useRef(null);

  const API_URL = "https://health-workers-assistant-project.onrender.com//query";

  // store last query to avoid stale updates
  const lastQueryRef = useRef("");

  // -------------------------------------------------------
  // MAIN SUBMIT HANDLER
  // -------------------------------------------------------
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuestion = input.trim();
    setCurrentQuery(userQuestion);
    lastQueryRef.current = userQuestion;

    setLoadingTable(true);
    setLoadingInsights(false);
    setError("");

    // reset UI
    setResult(null);
    setInput("");
    setCurrentPage(1);
    setInsightsAvailable(false);
    setShowCharts(false);

    try {

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          debug: false,
          page: 1,
          page_size: rowsPerPage,
          include_rows: true,
          include_insights: false,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Something went wrong");

      setResult({
        question: userQuestion,
        view: data.view || "records",
        columns: data.columns || [],
        rows: data.rows || [],
        row_count: data.row_count ?? (data.rows ? data.rows.length : 0),
        insights: null,
      });

      if (data.insights_available) {
        setLoadingInsights(true);

        fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: userQuestion,
            debug: false,
            page: 1,
            page_size: rowsPerPage,
            include_rows: false,
            include_insights: true,
          }),
        })
          .then((r) => r.json())
          .then((insightData) => {
            if (lastQueryRef.current !== userQuestion) return;

            if (insightData?.insights) {
              setResult((prev) =>
                prev
                  ? {
                      ...prev,
                      insights: insightData.insights,
                      row_count: prev.row_count,
                    }
                  : prev
              );

              setInsightsAvailable(true);
              setShowCharts(true);
            }
          })
          .catch((err) => {
            console.warn("Insights fetch failed:", err);
          })
          .finally(() => {
            if (lastQueryRef.current === userQuestion) {
              setLoadingInsights(false);
            }
          });
      }
    } catch (err) {
      setError(err.message || "Failed to fetch API");
      setInsightsAvailable(false);
      setShowCharts(false);
    } finally {
      setLoadingTable(false);
    }
  };

  // -------------------------------------------------------
  // PAGINATION (ONLY FOR RECORDS VIEW)
  // -------------------------------------------------------
  useEffect(() => {
    if (!currentQuery) return;

    // Pagination should NOT run for summary mode
    if (result?.view === "summary") return;

    const controller = new AbortController();

    // cancel previous request immediately
    if (pageAbortRef.current) {
      pageAbortRef.current.abort();
    }

    pageAbortRef.current = controller;

    const fetchPage = async () => {
      setLoadingTable(true);
      setError("");

      try {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            question: currentQuery,
            debug: false,
            page: currentPage,
            page_size: rowsPerPage,
            include_rows: true,
            include_insights: false,
          }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || "Something went wrong");

        setResult((prev) =>
          prev
            ? {
                ...prev,
                rows: data.rows || [],
                row_count: data.row_count ?? prev.row_count,
              }
            : prev
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Failed to fetch API");
        }
      } finally {
        setLoadingTable(false);
      }
    };

    fetchPage();

    return () => controller.abort();
  }, [currentQuery, currentPage]);

  // -------------------------------------------------------
  // UI HELPERS
  // -------------------------------------------------------
  const paginatedRows = Array.isArray(result?.rows) ? result.rows : [];
  const totalPages = Math.ceil((result?.row_count || 0) / rowsPerPage);

  const canGoPrev = currentPage > 1;
  const canGoNext = currentPage < totalPages;

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // -------------------------------------------------------
  // RENDER
  // -------------------------------------------------------
  return (
    <div className="chat-container">
      <div className="chat-interface">
        {/* Input */}
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
              disabled={!input.trim()}
            >
              {loadingTable ? "..." : "Send"}
            </button>
          </div>
        </form>

        {/* Layout */}
        <div
          className={`content-grid ${
            showCharts && insightsAvailable ? "with-insights" : ""
          }`}
        >
          {/* Output */}
          <div className="output-section">
            {currentQuery && (
              <div className="query-display">
                <span className="query-label">Query:</span>
                <div className="query-text">{currentQuery}</div>
              </div>
            )}

            <div className="output-box">
              {error && <div className="error-text">Error: {error}</div>}
              {loadingTable && (
                <div className="loading-text">Fetching results...</div>
              )}

              {!loadingTable && result && result.columns.length > 0 && (
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
                      {paginatedRows.length > 0 ? (
                        paginatedRows.map((row, rIdx) => (
                          <tr key={rIdx}>
                            {result.columns.map((_, cIdx) => (
                              <td key={cIdx}>
                                {row[cIdx] === null || row[cIdx] === undefined
                                  ? "-"
                                  : typeof row[cIdx] === "number"
                                    ? Math.floor(row[cIdx]).toLocaleString()
                                    : String(row[cIdx])
                                }
                              </td>
                            ))}
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td
                            colSpan={result.columns.length}
                            style={{ padding: "14px" }}
                          >
                            No rows returned yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}

              {loadingInsights && (
                <div className="loading-text">Loading insights…</div>
              )}
            </div>

            {/* Pagination only for records */}
            {result?.view === "records" && (
              <div className="pagination-controls">
                <button
                  onClick={() => setCurrentPage((p) => p - 1)}
                  disabled={!canGoPrev}
                >
                  Previous
                </button>

                <span>
                  Page {currentPage} of {totalPages}
                </span>

                <button
                  onClick={() => setCurrentPage((p) => p + 1)}
                  disabled={!canGoNext}
                >
                  Next
                </button>
              </div>
            )}
          </div>

          {/* Charts */}
          {showCharts && insightsAvailable && (
            <div className="insights-section">
              <ChartsPanel insights={result?.insights} view={result?.view} />
            </div>
          )}
        </div>

        {/* Floating Button */}
        <div className="insights-float">
          <button
            className="insights-btn"
            disabled={!insightsAvailable}
            onClick={() => setShowCharts((p) => !p)}
          >
            {showCharts ? "Hide Insights" : "Insights"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
