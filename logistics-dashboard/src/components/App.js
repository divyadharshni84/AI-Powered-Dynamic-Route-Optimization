import React from 'react';
import './App.css';
import Sidebar from './Sidebar.js';
import Dashboard from './Dashboard';
import Map from './Map';

function App() {
  return (
    <div className="App" style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ flex: 1, padding: '20px', backgroundColor: '#f4f4f4' }}>
        <Dashboard />
        <Map />
      </div>
    </div>
  );
}

export default App;
