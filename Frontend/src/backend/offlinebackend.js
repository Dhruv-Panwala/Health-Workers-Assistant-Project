import { registerPlugin } from "@capacitor/core";

const offlineBackendCandidates = [
  registerPlugin("OfflineBackend"),
  registerPlugin("offlineBackend"),
];

function extractErrorMessage(errorLike) {
  if (!errorLike) return "";
  if (typeof errorLike === "string") return errorLike;
  if (typeof errorLike.detail === "string" && errorLike.detail.trim()) {
    return errorLike.detail;
  }
  if (typeof errorLike.message === "string" && errorLike.message.trim()) {
    return errorLike.message;
  }
  return String(errorLike);
}

export function getOfflineBackendErrorMessage(errorLike) {
  const rawMessage = extractErrorMessage(errorLike)
    .trim()
    .replace(/^Error:\s*/i, "");
  const errorCode =
    errorLike && typeof errorLike === "object" ? errorLike.error_code : "";

  if (!rawMessage) {
    return "Something went wrong while fetching the result. Please try again.";
  }

  if (errorCode === "summary_shape_error") {
    return "I couldn't summarize that question cleanly. Try rephrasing it or ask for the raw table.";
  }

  if (errorCode === "database_missing") {
    return "The offline database is missing on this device.";
  }

  if (errorCode === "empty_question") {
    return "Please enter a question before sending.";
  }

  if (errorCode === "query_failed") {
    return "I couldn't process that request. Please try again or rephrase the question.";
  }

  if (/^["']?[a-z_][a-z0-9_]*["']?$/i.test(rawMessage)) {
    return "I couldn't summarize that question cleanly. Try rephrasing it or ask for the raw table.";
  }

  if (/SQLite database not found/i.test(rawMessage)) {
    return "The offline database is missing on this device.";
  }

  if (/No question provided/i.test(rawMessage)) {
    return "Please enter a question before sending.";
  }

  if (/plugin is not implemented|is not implemented on/i.test(rawMessage)) {
    return "The offline backend is not available on this device yet.";
  }

  if (/No local model found|Failed to load model|Model initialization failed/i.test(rawMessage)) {
    return "The local AI model is not available on this device.";
  }

  return rawMessage;
}

function createOfflineBackendError(errorLike) {
  const rawMessage = extractErrorMessage(errorLike);
  const error = new Error(getOfflineBackendErrorMessage(errorLike));
  error.rawMessage = rawMessage;
  if (errorLike && typeof errorLike === "object") {
    error.errorCode = errorLike.error_code;
    error.technicalDetail = errorLike.technical_detail;
  }
  return error;
}

async function callOfflineBackend(method, payload) {
  let lastError;

  for (const plugin of offlineBackendCandidates) {
    try {
      return await plugin[method](payload);
    } catch (error) {
      const message = extractErrorMessage(error);
      if (!/plugin is not implemented|is not implemented on/i.test(message)) {
        throw createOfflineBackendError(error);
      }
      lastError = error;
    }
  }

  throw createOfflineBackendError(
    lastError ?? { detail: "offlineBackend plugin is not implemented" }
  );
}

export async function runQuery(payload) {
  const result = await callOfflineBackend("runQuery", {
    question: payload.question,
    debug: payload.debug ?? false,
    page: payload.page ?? 1,
    page_size: payload.page_size ?? 200,
    include_rows: payload.include_rows ?? true,
    include_insights: payload.include_insights ?? false,
  });

  if (result?.detail) {
    throw createOfflineBackendError(result);
  }

  return result;
}
