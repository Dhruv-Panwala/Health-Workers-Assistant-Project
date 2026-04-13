import { useEffect, useState } from "react";
import {
  getRuntimeBundleInfo,
  getRuntimeStatus,
  isNativeRuntime,
  prepareRuntime,
} from "../lib/runtimeClient";

function buildInitialRuntimeState(nativeRuntime) {
  return {
    ready: !nativeRuntime,
    preparing: nativeRuntime,
    error: "",
    copiedBytes: 0,
    totalBytes: 0,
    activeDb: "",
    loadedModel: null,
    modelsSupported: !nativeRuntime,
    sqlModelSupported: !nativeRuntime,
    chatModelSupported: !nativeRuntime,
    modelSupportError: "",
    sqlModelError: "",
    chatModelError: "",
  };
}

export default function useRuntimeState() {
  const nativeRuntime = isNativeRuntime();
  const [runtimeState, setRuntimeState] = useState(() =>
    buildInitialRuntimeState(nativeRuntime)
  );

  useEffect(() => {
    let cancelled = false;

    const bootstrapRuntime = async () => {
      if (!nativeRuntime) {
        return;
      }

      setRuntimeState((prev) => ({
        ...prev,
        preparing: true,
        error: "",
      }));

      try {
        const bundleInfo = await getRuntimeBundleInfo();

        if (cancelled) {
          return;
        }

        setRuntimeState((prev) => ({
          ...prev,
          totalBytes: bundleInfo.totalBytes ?? prev.totalBytes,
        }));

        const prepareResult = await prepareRuntime();
        const status = await getRuntimeStatus();

        if (cancelled) {
          return;
        }

        setRuntimeState({
          ready: Boolean(prepareResult.ready),
          preparing: false,
          error: "",
          copiedBytes: prepareResult.copiedBytes ?? 0,
          totalBytes: prepareResult.totalBytes ?? 0,
          activeDb: prepareResult.activeDb ?? "",
          loadedModel: status.loadedModel ?? null,
          modelsSupported: status.modelsSupported ?? true,
          sqlModelSupported: status.sqlModelSupported ?? true,
          chatModelSupported: status.chatModelSupported ?? true,
          sqlModelError: status.sqlModelError ?? "",
          chatModelError: status.chatModelError ?? "",
          modelSupportError: status.modelSupportError ?? "",
        });
      } catch (runtimeError) {
        if (cancelled) {
          return;
        }

        setRuntimeState((prev) => ({
          ...prev,
          preparing: false,
          ready: false,
          error: runtimeError.message || "Offline runtime preparation failed.",
        }));
      }
    };

    bootstrapRuntime();

    return () => {
      cancelled = true;
    };
  }, [nativeRuntime]);

  const refreshRuntimeStatus = async () => {
    if (!nativeRuntime) {
      return;
    }

    try {
      const status = await getRuntimeStatus();
      setRuntimeState((prev) => ({
        ...prev,
        ready: Boolean(status.assetsReady),
        loadedModel: status.loadedModel ?? null,
        modelsSupported: status.modelsSupported ?? prev.modelsSupported,
        sqlModelSupported: status.sqlModelSupported ?? prev.sqlModelSupported,
        chatModelSupported: status.chatModelSupported ?? prev.chatModelSupported,
        sqlModelError: status.sqlModelError ?? prev.sqlModelError,
        chatModelError: status.chatModelError ?? prev.chatModelError,
        modelSupportError: status.modelSupportError ?? prev.modelSupportError,
      }));
    } catch {
      // Ignore secondary status failures after a request finishes.
    }
  };

  return {
    nativeRuntime,
    runtimeState,
    refreshRuntimeStatus,
  };
}
