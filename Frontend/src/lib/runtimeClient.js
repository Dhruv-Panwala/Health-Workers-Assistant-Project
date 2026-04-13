import { Capacitor } from "@capacitor/core";
import { UlizaEngine } from "../plugins/ulizaEngine";

const WEB_API_URL =
  import.meta.env.VITE_API_URL ??
  "https://health-worker-assistant-project.onrender.com/query";

export function isNativeRuntime() {
  return Capacitor.isNativePlatform();
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;

  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }

  return `${value.toFixed(value >= 100 || index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatGigabytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0.00 GB";
  }

  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export async function prepareRuntime() {
  if (!isNativeRuntime()) {
    return {
      ready: true,
      copiedBytes: 0,
      totalBytes: 0,
      activeDb: "remote",
    };
  }

  return UlizaEngine.prepareAssets();
}

export async function getRuntimeBundleInfo() {
  if (!isNativeRuntime()) {
    return {
      totalBytes: 0,
    };
  }

  return UlizaEngine.getBundleInfo();
}

export async function getRuntimeStatus() {
  if (!isNativeRuntime()) {
    return {
      assetsReady: true,
      loadedModel: null,
      modelsSupported: true,
      sqlModelSupported: true,
      chatModelSupported: true,
      sqlModelError: null,
      chatModelError: null,
      modelSupportError: null,
    };
  }

  return UlizaEngine.getRuntimeStatus();
}

export async function executeQueryRequest(payload) {
  if (isNativeRuntime()) {
    return UlizaEngine.query(payload);
  }

  const response = await fetch(WEB_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || "Something went wrong");
  }

  return data;
}

export async function executeChatConversation(payload) {
  if (isNativeRuntime()) {
    return UlizaEngine.chatConversation(payload);
  }

  throw new Error("CURE-MED Chat is only available on Android.");
}

export async function prewarmChatModel() {
  if (isNativeRuntime()) {
    return UlizaEngine.prewarmChatModel();
  }

  return {
    ready: true,
    loadedModel: "chat",
  };
}

export async function unloadRuntimeModel() {
  if (isNativeRuntime()) {
    return UlizaEngine.unloadModel();
  }

  return {
    loadedModel: null,
  };
}
