import React, { useState, useEffect } from 'react';
import { Download, X, Smartphone } from 'lucide-react';
import './InstallPrompt.css';

export default function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    const handleBeforeInstallPrompt = (e) => {
      // Prevent browser's automatic default mini-infobar
      e.preventDefault();
      setDeferredPrompt(e);
      // Check if user previously dismissed it in this session
      const dismissed = sessionStorage.getItem('pwa_install_dismissed');
      if (!dismissed) {
        setShowPrompt(true);
      }
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    console.log('[PWA Install Outcome]:', outcome);
    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  const handleDismiss = () => {
    setShowPrompt(false);
    sessionStorage.setItem('pwa_install_dismissed', 'true');
  };

  if (!showPrompt) return null;

  return (
    <div className="install-prompt-banner" role="alert" aria-live="polite">
      <div className="install-prompt-content">
        <div className="install-icon-wrapper">
          <Smartphone className="install-device-icon" />
        </div>
        <div className="install-text-wrapper">
          <span className="install-title">Install LeafScan App</span>
          <span className="install-subtitle">Add to home screen for fast mobile access</span>
        </div>
      </div>
      <div className="install-actions">
        <button 
          className="install-btn"
          onClick={handleInstallClick}
          aria-label="Install LeafScan application"
        >
          <Download className="btn-icon" />
          <span>Install</span>
        </button>
        <button 
          className="dismiss-btn"
          onClick={handleDismiss}
          aria-label="Dismiss install banner"
        >
          <X className="btn-icon" />
        </button>
      </div>
    </div>
  );
}
