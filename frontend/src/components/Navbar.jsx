import React, { useState, useEffect } from 'react';
import { Leaf, Activity, RefreshCw, Menu, X } from 'lucide-react';
import './Navbar.css';

/**
 * Navbar Component (Step 9 Updated)
 * --------------------------------
 * Provides top navigation bar with project logo, section navigation links,
 * mobile responsive drawer toggle, and live Flask backend API connection status monitor.
 */
export default function Navbar() {
  const [apiStatus, setApiStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [apiMessage, setApiMessage] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const checkHealth = async () => {
    setIsRefreshing(true);
    const apiBase = import.meta.env.VITE_API_URL || '';
    try {
      const res = await fetch(`${apiBase}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setApiStatus('online');
        setApiMessage(data.message || 'API connected');
      } else {
        setApiStatus('offline');
        setApiMessage('HTTP Error ' + res.status);
      }
    } catch (err) {
      setApiStatus('offline');
      setApiMessage('Cannot reach Flask server (Port 5000)');
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
  };

  return (
    <header className="navbar-header">
      <div className="navbar-container">
        {/* Brand / Logo */}
        <a href="#home" className="navbar-brand" onClick={closeMobileMenu}>
          <div className="brand-icon">
            <Leaf className="leaf-icon" />
          </div>
          <div className="brand-text">
            <span className="brand-title">LeafScan</span>
            <span className="brand-tag">Deep Learning AI</span>
          </div>
        </a>

        {/* Desktop Navigation Links */}
        <nav className="navbar-nav desktop-nav">
          <a href="#home" className="nav-link">Home</a>
          <a href="#upload" className="nav-link">Upload</a>
          <a href="#about" className="nav-link">About</a>
          <a href="#how-it-works" className="nav-link">How It Works</a>
        </nav>

        {/* Right Section: API Badge + Refresh + Mobile Menu Toggle */}
        <div className="navbar-actions">
          <div 
            className={`api-status-badge ${apiStatus}`}
            title={apiMessage}
          >
            <span className={`pulse-dot ${apiStatus === 'offline' ? 'offline' : ''}`}></span>
            <span className="status-text">
              {apiStatus === 'checking' && 'Checking API...'}
              {apiStatus === 'online' && 'API Online'}
              {apiStatus === 'offline' && 'API Offline'}
            </span>
          </div>
          
          <button 
            className="refresh-btn" 
            onClick={checkHealth} 
            disabled={isRefreshing}
            title="Refresh Backend Status"
            aria-label="Refresh Backend Status"
          >
            <RefreshCw className={`refresh-icon ${isRefreshing ? 'spin' : ''}`} />
          </button>

          {/* Mobile Menu Hamburger Button */}
          <button
            className="mobile-menu-toggle"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle navigation menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="menu-icon" /> : <Menu className="menu-icon" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Dropdown Menu */}
      {mobileMenuOpen && (
        <div className="mobile-menu-drawer">
          <nav className="mobile-nav-links">
            <a href="#home" className="mobile-nav-link" onClick={closeMobileMenu}>Home</a>
            <a href="#upload" className="mobile-nav-link" onClick={closeMobileMenu}>Upload Leaf</a>
            <a href="#about" className="mobile-nav-link" onClick={closeMobileMenu}>About AI Model</a>
            <a href="#how-it-works" className="mobile-nav-link" onClick={closeMobileMenu}>How It Works</a>
          </nav>
        </div>
      )}
    </header>
  );
}
