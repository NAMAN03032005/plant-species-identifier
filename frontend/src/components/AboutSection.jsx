import React from 'react';
import { BookOpen, Layers, Zap, Code2, Cpu, CheckCircle, Smartphone } from 'lucide-react';
import './AboutSection.css';

/**
 * AboutSection Component (Step 9 Updated)
 * ---------------------------------------
 * Explains academic project scope, transfer learning concepts, system architecture,
 * and technical MobileNetV2 Keras web engine vs. TensorFlow Lite mobile model specifications.
 */
export default function AboutSection() {
  return (
    <section id="about" className="section">
      <h2 className="section-title">About LeafScan</h2>
      
      {/* Prompt Statement Box */}
      <div className="about-main-quote glass-card">
        <BookOpen className="quote-icon" />
        <p className="quote-text">
          "LeafScan is a deep learning based plant species identification application that uses transfer learning to classify plant leaves from images."
        </p>
      </div>

      <div className="about-grid">
        {/* Card 1: Transfer Learning & MobileNetV2 */}
        <div className="about-card glass-card">
          <div className="card-header">
            <div className="card-icon-box">
              <Cpu className="card-icon" />
            </div>
            <h3>Why Transfer Learning?</h3>
          </div>
          <p>
            Training deep neural networks from scratch requires massive datasets and high compute power. 
            Transfer Learning repurposes a pre-trained feature extractor (<strong>MobileNetV2</strong>, trained on ImageNet) 
            and fine-tunes top classification layers for specialized plant species identification.
          </p>
          <ul className="about-bullets">
            <li><CheckCircle className="bullet-icon" /> High classification accuracy with smaller training datasets</li>
            <li><CheckCircle className="bullet-icon" /> Dramatically faster training convergence</li>
            <li><CheckCircle className="bullet-icon" /> Reduced computational resource requirement</li>
          </ul>
        </div>

        {/* Card 2: MobileNetV2 Efficiency */}
        <div className="about-card glass-card">
          <div className="card-header">
            <div className="card-icon-box">
              <Zap className="card-icon" />
            </div>
            <h3>MobileNetV2 Efficiency</h3>
          </div>
          <p>
            MobileNetV2 introduces inverted residual blocks and depthwise separable convolutions, making it ideal 
            for lightweight web and mobile deployment without sacrificing model precision.
          </p>
          <ul className="about-bullets">
            <li><CheckCircle className="bullet-icon" /> Low latency inference for real-time web feedback</li>
            <li><CheckCircle className="bullet-icon" /> Compact weight size (<strong>9.34 MB Keras / 8.53 MB TFLite</strong>)</li>
            <li><CheckCircle className="bullet-icon" /> Mobile browser and edge-device friendly</li>
          </ul>
        </div>

        {/* Card 3: Full-Stack Architecture */}
        <div className="about-card glass-card">
          <div className="card-header">
            <div className="card-icon-box">
              <Code2 className="card-icon" />
            </div>
            <h3>Full-Stack System Architecture</h3>
          </div>
          <p>
            The project follows a clean client-server separation of concerns:
          </p>
          <ul className="about-bullets">
            <li><CheckCircle className="bullet-icon" /> <strong>Frontend</strong>: React + Vite responsive Progressive Web App</li>
            <li><CheckCircle className="bullet-icon" /> <strong>Backend</strong>: Python Flask microservice REST API</li>
            <li><CheckCircle className="bullet-icon" /> <strong>Communication</strong>: Asynchronous HTTP multipart requests</li>
          </ul>
        </div>

        {/* Card 4: AI Model Specifications */}
        <div className="about-card glass-card model-specs-card">
          <div className="card-header">
            <div className="card-icon-box">
              <Smartphone className="card-icon" />
            </div>
            <h3>AI Model Specifications</h3>
          </div>
          <div className="specs-table">
            <div className="spec-row">
              <span className="spec-label">Architecture</span>
              <span className="spec-value">MobileNetV2</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Transfer Learning</span>
              <span className="spec-value">Yes (ImageNet Weights)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Web Engine</span>
              <span className="spec-value">TensorFlow / Keras (Flask API)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Mobile Engine</span>
              <span className="spec-value">TensorFlow Lite (.tflite Ready)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Input Tensor</span>
              <span className="spec-value">224 × 224 × 3 (RGB)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Total Parameters</span>
              <span className="spec-value">2,270,794</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
