import React, { useEffect, useRef, useState } from "react";
import {
  executeChatConversation,
  prewarmChatModel,
  unloadRuntimeModel,
} from "../lib/runtimeClient";
import "./CureMedChat.css";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "ENGLISH" },
  { value: "sw", label: "Kiswahili" },
];

function CureMedChat({ isActive, nativeRuntime, runtimeState, refreshRuntimeStatus }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("en");
  const [loading, setLoading] = useState(false);
  const [warmingUp, setWarmingUp] = useState(false);
  const [error, setError] = useState("");
  const transcriptEndRef = useRef(null);
  const hasPrewarmedRef = useRef(false);

  const runtimeBusy = nativeRuntime && (runtimeState.preparing || !runtimeState.ready);
  const chatLoading = nativeRuntime && runtimeState.ready && (warmingUp || runtimeState.loadedModel !== "chat");
  const chatUnavailable =
    nativeRuntime &&
    !runtimeState.preparing &&
    !runtimeState.error &&
    runtimeState.ready &&
    !runtimeState.chatModelSupported;
  const statusMessage = runtimeState.error
    ? runtimeState.error
    : runtimeBusy
      ? "Preparing offline runtime for CURE-MED Chat..."
      : chatLoading
        ? "Loading CURE-MED Chat model..."
      : chatUnavailable
        ? runtimeState.chatModelError || "CURE-MED Chat is not available on this device."
        : "";

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    if (!isActive || !nativeRuntime || runtimeBusy || chatUnavailable) {
      return;
    }
    if (runtimeState.loadedModel === "chat") {
      hasPrewarmedRef.current = true;
      setWarmingUp(false);
      return;
    }
    if (hasPrewarmedRef.current || warmingUp) {
      return;
    }

    let cancelled = false;
    setWarmingUp(true);
    setError("");

    (async () => {
      try {
        await prewarmChatModel();
        await refreshRuntimeStatus();
        if (!cancelled) {
          hasPrewarmedRef.current = true;
        }
      } catch (chatError) {
        if (!cancelled) {
          setError(chatError.message || "CURE-MED model failed to load.");
        }
      } finally {
        if (!cancelled) {
          setWarmingUp(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    chatUnavailable,
    isActive,
    nativeRuntime,
    refreshRuntimeStatus,
    runtimeBusy,
    runtimeState.loadedModel,
    warmingUp,
  ]);

  useEffect(() => {
    if (!nativeRuntime || runtimeBusy) {
      return;
    }

    let cancelled = false;

    const syncActiveModel = async () => {
      const loadedModel = runtimeState.loadedModel;
      if (isActive && loadedModel && loadedModel !== "chat") {
        try {
          await unloadRuntimeModel();
          if (!cancelled) {
            await refreshRuntimeStatus();
            hasPrewarmedRef.current = false;
          }
        } catch {
          // The prewarm path will surface any remaining runtime issue.
        }
      }

      if (!isActive && loadedModel === "chat") {
        try {
          await unloadRuntimeModel();
          if (!cancelled) {
            await refreshRuntimeStatus();
            hasPrewarmedRef.current = false;
          }
        } catch {
          // Ignore unload failures during tab switches.
        }
      }
    };

    syncActiveModel();

    return () => {
      cancelled = true;
    };
  }, [
    isActive,
    nativeRuntime,
    refreshRuntimeStatus,
    runtimeBusy,
    runtimeState.loadedModel,
  ]);

  const submitMessage = async (event) => {
    event.preventDefault();
    if (!input.trim() || loading || warmingUp || runtimeBusy || chatUnavailable) {
      return;
    }

    const userMessage = {
      role: "user",
      content: input.trim(),
    };
    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);
    setInput("");
    setError("");
    setLoading(true);

    try {
      const response = await executeChatConversation({
        messages: nextMessages,
        language,
      });
      await refreshRuntimeStatus();

      const reply = response?.reply?.trim();
      if (!reply) {
        throw new Error("CURE-MED returned an empty reply.");
      }

      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: reply,
        },
      ]);
    } catch (chatError) {
      setError(chatError.message || "CURE-MED Chat failed.");
      await refreshRuntimeStatus();
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInput("");
    setError("");
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitMessage(event);
    }
  };

  return (
    <div className="cure-med-chat">
      <div className="cure-med-toolbar">
        <div className="cure-med-toolbar-group">
          <label className="cure-med-label" htmlFor="chat-language">
            Reply language
          </label>
          <select
            id="chat-language"
            className="cure-med-select"
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
          disabled={loading || warmingUp}
          >
            {LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          className="cure-med-clear"
          onClick={clearChat}
          disabled={!messages.length && !input}
        >
          Clear Chat
        </button>
      </div>

      {statusMessage && (
        <div className="cure-med-status">{statusMessage}</div>
      )}

      <div className="cure-med-transcript">
        {!messages.length && !loading && (
          <div className="cure-med-empty">
            Start a conversation with the offline CURE-MED assistant.
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`cure-med-bubble cure-med-bubble-${message.role}`}
          >
            <div className="cure-med-role">
              {message.role === "assistant" ? "CURE-MED" : "You"}
            </div>
            <div className="cure-med-content">{message.content}</div>
          </div>
        ))}

        {loading && (
          <div className="cure-med-bubble cure-med-bubble-assistant">
            <div className="cure-med-role">CURE-MED</div>
            <div className="cure-med-content">Thinking...</div>
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>

      {error && <div className="cure-med-error">Error: {error}</div>}

      <form className="cure-med-form" onSubmit={submitMessage}>
        <textarea
          className="cure-med-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            runtimeBusy
              ? "Preparing offline runtime for this device..."
              : "Ask CURE-MED a health question..."
          }
          rows="4"
          disabled={loading || warmingUp || runtimeBusy || chatUnavailable}
        />
        <button
          type="submit"
          className="cure-med-send"
          disabled={!input.trim() || loading || warmingUp || runtimeBusy || chatUnavailable}
        >
          {loading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}

export default CureMedChat;
