// src/App.js
import React from 'react';
import AudioVADClient from './components/AudioVADClient';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <AudioVADClient />
      </header>
    </div>
  );
}

export default App;
