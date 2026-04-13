import React, { useState } from "react";
import Header from "./components/Header.jsx";
import ChatInterface from "./components/ChatInterface.jsx";
import CureMedChat from "./components/CureMedChat.jsx";
import LearningButton from "./components/LearningButton.jsx";
import useRuntimeState from "./hooks/useRuntimeState.js";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("dataAssistant");
  const { nativeRuntime, runtimeState, refreshRuntimeStatus } = useRuntimeState();

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="app-shell">
          <div className="app-tabs" role="tablist" aria-label="Assistant modes">
            <button
              type="button"
              role="tab"
              id="tab-data-assistant"
              aria-controls="panel-data-assistant"
              aria-selected={activeTab === "dataAssistant"}
              className={`app-tab ${activeTab === "dataAssistant" ? "app-tab-active" : ""}`}
              onClick={() => setActiveTab("dataAssistant")}
            >
              DataAssistant
            </button>

            {nativeRuntime && (
              <button
                type="button"
                role="tab"
                id="tab-cure-med-chat"
                aria-controls="panel-cure-med-chat"
                aria-selected={activeTab === "cureMedChat"}
                className={`app-tab ${activeTab === "cureMedChat" ? "app-tab-active" : ""}`}
                onClick={() => setActiveTab("cureMedChat")}
              >
                CURE-MED Chat
              </button>
            )}
          </div>

          <div className="app-panels">
            <section
              id="panel-data-assistant"
              role="tabpanel"
              aria-labelledby="tab-data-assistant"
              className="app-panel"
              hidden={activeTab !== "dataAssistant"}
            >
              <ChatInterface
                nativeRuntime={nativeRuntime}
                runtimeState={runtimeState}
                refreshRuntimeStatus={refreshRuntimeStatus}
              />
            </section>

            {nativeRuntime && (
              <section
                id="panel-cure-med-chat"
                role="tabpanel"
                aria-labelledby="tab-cure-med-chat"
                className="app-panel"
                hidden={activeTab !== "cureMedChat"}
              >
                <CureMedChat
                  isActive={activeTab === "cureMedChat"}
                  nativeRuntime={nativeRuntime}
                  runtimeState={runtimeState}
                  refreshRuntimeStatus={refreshRuntimeStatus}
                />
              </section>
            )}
          </div>
        </div>
      </main>
      <LearningButton />
    </div>
  );
}

export default App;
