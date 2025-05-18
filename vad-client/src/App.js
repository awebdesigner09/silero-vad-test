import React from 'react';
import AudioVADClient from './components/AudioVADClient'; // Updated path
import './App.css'; // Optional: if you have App-specific CSS

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