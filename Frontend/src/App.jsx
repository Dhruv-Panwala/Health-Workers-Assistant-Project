import React from 'react'
import Header from './components/Header.jsx'
import ChatInterface from './components/ChatInterface.jsx'
import LearningButton from './components/LearningButton.jsx'
import './App.css'
import ChartsPanel from './components/ChartPanel.jsx'

function App() {
  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <ChatInterface />
      </main>
      <LearningButton />
    </div>
  )
}

export default App
