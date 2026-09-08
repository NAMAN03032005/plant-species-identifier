import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, Cpu, Upload } from 'lucide-react';
import './Hero.css';

/**
 * Hero Component (Step 2 Updated)
 * --------------------------------
 * Displays main heading "Identify Plants from Their Leaves", educational overview
 * of deep learning transfer learning, and CTA scrolling directly to upload section.
 */
export default function Hero() {
  const scrollToUpload = (e) => {
    e.preventDefault();
    const uploadElem = document.getElementById('upload');
    if (uploadElem) {
      uploadElem.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="home" className="hero-section">
      <div className="hero-container">
        {/* Academic Project Badge */}
        <div className="hero-pill">
          <Sparkles className="pill-icon" />
          <span>Academic Deep Learning Project • Step 2 Upload & Validation</span>
        </div>

        {/* Main Heading */}
        <h1 className="hero-title">
          Identify Plants from Their Leaves
        </h1>
        
        <h2 className="hero-subtitle">
          Plant Species Identification using Deep Learning & Transfer Learning
        </h2>

        {/* Supporting Copy */}
        <p className="hero-description">
          LeafScan uses fine-tuned Transfer Learning (MobileNetV2) to accurately classify plant species 
          from leaf photographs. Upload a leaf image below to validate your input and prepare for AI inference.
        </p>

        {/* Call to Action Buttons */}
        <div className="hero-actions">
          <a href="#upload" className="btn btn-primary" onClick={scrollToUpload}>
            <Upload className="btn-icon" /> Select Leaf Image
          </a>
          <a href="#how-it-works" className="btn btn-secondary">
            How It Works
          </a>
        </div>

        {/* Highlights */}
        <div className="hero-highlights">
          <div className="highlight-item">
            <Cpu className="highlight-icon" />
            <div>
              <strong>Transfer Learning</strong>
              <span>MobileNetV2 Neural Network</span>
            </div>
          </div>

          <div className="highlight-item">
            <ShieldCheck className="highlight-icon" />
            <div>
              <strong>Validation API</strong>
              <span>Python Flask Microservice</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
