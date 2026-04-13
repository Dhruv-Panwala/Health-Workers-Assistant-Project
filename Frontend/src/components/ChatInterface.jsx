import React, { useEffect, useRef, useState } from "react";
import ChartsPanel from "./ChartPanel";
import {
  executeQueryRequest,
  formatGigabytes,
} from "../lib/runtimeClient";
import "./ChatInterface.css";

function ChatInterface({ nativeRuntime, runtimeState, refreshRuntimeStatus }) {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [resolvedPlan, setResolvedPlan] = useState(null);
  const [loadingTable, setLoadingTable] = useState(false);
  const [loadingInsights, setLoadingInsights] = useState(false);
  const [error, setError] = useState("");
  const [insightError, setInsightError] = useState("");
  const [insightsAvailable, setInsightsAvailable] = useState(false);
  const [showCharts, setShowCharts] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(100);

  const pageAbortRef = useRef(null);
  const insightsAbortRef = useRef(null);
  const lastQueryRef = useRef("");

  const runQueryRequest = async (payload) => {
    const data = await executeQueryRequest(payload);
    await refreshRuntimeStatus();
    return data;
  };

  const fetchDeferredInsights = async (queryText, nextResolvedPlan) => {
    if (!nextResolvedPlan) {
      return;
    }

    const controller = new AbortController();
    if (insightsAbortRef.current) {
      insightsAbortRef.current.abort();
    }
    insightsAbortRef.current = controller;

    setLoadingInsights(true);
    setInsightError("");

    try {
      const data = await runQueryRequest({
        question: queryText,
        resolved_plan: JSON.stringify(nextResolvedPlan),
        debug: false,
        page: 1,
        page_size: rowsPerPage,
        include_rows: false,
        include_insights: true,
      });

      if (controller.signal.aborted || lastQueryRef.current !== queryText) {
        return;
      }

      const hasSupportedInsights = Boolean(
        data?.insights_available &&
        data?.insights &&
        data.insights.mode !== "none"
      );

      setResult((prev) =>
        prev
          ? {
              ...prev,
              insights: data.insights || null,
            }
          : prev
      );
      setInsightsAvailable(hasSupportedInsights);
      setShowCharts(hasSupportedInsights);
    } catch (requestError) {
      if (controller.signal.aborted || lastQueryRef.current !== queryText) {
        return;
      }
      setInsightError(
        requestError.message || "Insights unavailable for this query."
      );
      setInsightsAvailable(false);
      setShowCharts(false);
    } finally {
      if (!controller.signal.aborted && lastQueryRef.current === queryText) {
        setLoadingInsights(false);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || (nativeRuntime && !runtimeState.ready)) return;

    const userQuestion = input.trim();
    pageAbortRef.current?.abort();
    insightsAbortRef.current?.abort();
    setCurrentQuery(userQuestion);
    lastQueryRef.current = userQuestion;

    setLoadingTable(true);
    setLoadingInsights(false);
    setError("");
    setInsightError("");

    setResult(null);
    setResolvedPlan(null);
    setInput("");
    setCurrentPage(1);
    setInsightsAvailable(false);
    setShowCharts(false);

    try {
      const data = await runQueryRequest({
        question: userQuestion,
        debug: false,
        page: 1,
        page_size: rowsPerPage,
        include_rows: true,
        include_insights: false,
      });
      if (lastQueryRef.current !== userQuestion) return;

      setResult({
        question: userQuestion,
        view: data.view || "records",
        columns: data.columns || [],
        rows: data.rows || [],
        row_count: data.row_count ?? (data.rows ? data.rows.length : 0),
        answer: data.answer || null,
        insights: data.insights || null,
      });
      setResolvedPlan(data.resolved_plan || null);
      setInsightsAvailable(false);
      setShowCharts(false);
      if (data.resolved_plan) {
        fetchDeferredInsights(userQuestion, data.resolved_plan);
      }
    } catch (requestError) {
      setError(requestError.message || "Failed to fetch API");
      setInsightsAvailable(false);
      setShowCharts(false);
    } finally {
      setLoadingTable(false);
      if (lastQueryRef.current === userQuestion) {
        setLoadingInsights(false);
      }
    }
  };

  useEffect(() => {
    if (!currentQuery || currentPage <= 1 || (nativeRuntime && !runtimeState.ready))
      return;
    if (result?.view !== "records") return;
    if (!resolvedPlan) return;

    const controller = new AbortController();

    if (pageAbortRef.current) {
      pageAbortRef.current.abort();
    }
    pageAbortRef.current = controller;

    const fetchPage = async () => {
      setLoadingTable(true);
      setError("");

      try {
        const data = await runQueryRequest({
          question: currentQuery,
          resolved_plan: JSON.stringify(resolvedPlan),
          debug: false,
          page: currentPage,
          page_size: rowsPerPage,
          include_rows: true,
          include_insights: false,
        });

        if (controller.signal.aborted) return;

        setResult((prev) =>
          prev
            ? {
                ...prev,
                rows: data.rows || [],
                row_count: data.row_count ?? prev.row_count,
              }
            : prev
        );
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          setError(requestError.message || "Failed to fetch API");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingTable(false);
        }
      }
    };

    fetchPage();

    return () => controller.abort();
  }, [
    currentQuery,
    currentPage,
    nativeRuntime,
    resolvedPlan,
    rowsPerPage,
    runtimeState.ready,
    result?.view,
  ]);

  useEffect(() => {
    return () => {
      pageAbortRef.current?.abort();
      insightsAbortRef.current?.abort();
    };
  }, []);

  const paginatedRows = Array.isArray(result?.rows) ? result.rows : [];
  const totalPages = Math.ceil((result?.row_count || 0) / rowsPerPage);
  const runtimeBusy = nativeRuntime && (runtimeState.preparing || !runtimeState.ready);
  const runtimeModelIssue =
    nativeRuntime &&
    !runtimeState.preparing &&
    !runtimeState.error &&
    runtimeState.ready &&
    !runtimeState.modelsSupported;
  const showRuntimeBanner =
    nativeRuntime &&
    (Boolean(runtimeState.error) || runtimeState.preparing || runtimeModelIssue);

  const canGoPrev = currentPage > 1;
  const canGoNext = currentPage < totalPages;

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-interface">
        {showRuntimeBanner && (
          <div
            className={`runtime-banner ${
              runtimeState.error
                ? "runtime-banner-error"
                : runtimeModelIssue
                  ? "runtime-banner-warning"
                : ""
            }`}
          >
            <div className="runtime-title">
              {runtimeState.error
                ? "Offline runtime issue"
                : runtimeState.preparing
                  ? "Preparing offline assets"
                  : runtimeModelIssue
                    ? "Offline runtime limited"
                  : ""}
            </div>

            <div className="runtime-description">
              {runtimeState.error
                ? runtimeState.error
                : runtimeState.preparing
                  ? runtimeState.totalBytes > 0
                    ? `Preparing to copy ${formatGigabytes(
                        runtimeState.totalBytes
                      )} of bundled models and databases to this device.`
                    : "Calculating bundled models and databases size..."
                  : runtimeModelIssue
                    ? "Bundled assets are ready, but one of the offline models did not load correctly on this device. Please restart the app or reinstall the APK if requests fail."
                  : ""}
            </div>
          </div>
        )}

        <form className="input-section input-top" onSubmit={handleSubmit}>
          <div className="input-container">
            <textarea
              className="input-box"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                runtimeBusy
                  ? "Preparing offline runtime for this device..."
                  : "Type your health-related question here..."
              }
              rows="3"
              disabled={runtimeBusy}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || runtimeBusy}
            >
              {loadingTable ? "..." : "Send"}
            </button>
          </div>
        </form>

        <div
          className={`content-grid ${
            showCharts && insightsAvailable ? "with-insights" : ""
          }`}
        >
          <div className="output-section">
            {currentQuery && (
              <div className="query-display">
                <span className="query-label">Query:</span>
                <div className="query-text">{currentQuery}</div>
              </div>
            )}

            <div className="output-box">
              {error && <div className="error-text">Error: {error}</div>}
              {insightError && !error && (
                <div className="error-text">Insights: {insightError}</div>
              )}
              {loadingTable && (
                <div className="loading-text">Fetching results...</div>
              )}

              {!loadingTable &&
                result?.view === "explainable" &&
                result?.answer && (
                  <div className="explainable-box">
                    <h3 style={{ marginBottom: "10px" }}>
                      Analytical Explanation
                    </h3>
                    <div style={{ whiteSpace: "pre-wrap" }}>{result.answer}</div>
                  </div>
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
                                    : String(row[cIdx])}
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
                <div className="loading-text">Loading insights...</div>
              )}
            </div>

            {result?.view === "records" && result?.columns?.length > 0 && (
              <div className="pagination-controls">
                <button
                  onClick={() => setCurrentPage((page) => page - 1)}
                  disabled={!canGoPrev}
                >
                  Previous
                </button>

                <span>
                  Page {currentPage} of {totalPages}
                </span>

                <button
                  onClick={() => setCurrentPage((page) => page + 1)}
                  disabled={!canGoNext}
                >
                  Next
                </button>
              </div>
            )}
          </div>

          {showCharts && insightsAvailable && (
            <div className="insights-section">
              <ChartsPanel insights={result?.insights} view={result?.view} />
            </div>
          )}
        </div>

        <div className="insights-float">
          <button
            className="insights-btn"
            disabled={!insightsAvailable}
            onClick={() => setShowCharts((visible) => !visible)}
          >
            {showCharts ? "Hide Insights" : "Insights"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
