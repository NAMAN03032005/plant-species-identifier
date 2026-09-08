import React from 'react';
import { Cpu, AlertCircle, CheckCircle2, BarChart2, BookOpen } from 'lucide-react';
import './ResultPlaceholder.css';

/**
 * ResultPlaceholder Component
 * ---------------------------
 * Displays clean placeholder state and mock result UI structure where 
 * Deep Learning classification predictions (species, confidence score) 
 * will be rendered after Step 2 model backend integration.
 */
export default function ResultPlaceholder({ selectedImage }) {
  return (
    <section id="results" className="section">
      <h2 className="section-title">Prediction Results</h2>
      <p className="section-subtitle">
        Deep Learning classification output, predicted species class, and confidence score display area.
      </p>

      <div className="results-container glass-card">
        {!selectedImage ? (
          /* Empty State */
          <div className="empty-results-state">
            <div className="empty-icon-box">
              <Cpu className="empty-icon" />
            </div>
            <h3 className="empty-title">Prediction Output Zone</h3>
            <p className="empty-description">
              Upload a plant leaf image in the section above to activate the preview layout.
            </p>
          </div>
        ) : (
          /* Structured Placeholder for Step 2 Model Results */
          <div className="results-preview-content">
            <div className="results-badge-header">
              <span className="step2-badge">
                <AlertCircle className="badge-icon" /> Step 2 AI Model Pipeline Pending
              </span>
              <span className="model-info-tag">Target Architecture: MobileNetV2 (Keras)</span>
            </div>

            <div className="result-grid">
              {/* Left Column: Image Thumbnail */}
              <div className="result-image-box">
                <img src={selectedImage.previewUrl} alt="Uploaded leaf" className="result-thumb" />
                <span className="image-caption">Uploaded Input Image</span>
              </div>

              {/* Right Column: Species Metadata Placeholder */}
              <div className="result-data-box">
                <div className="species-header-placeholder">
                  <div className="placeholder-tag">Species Name (Step 2)</div>
                  <h3 className="mock-species-title">Target Plant Species Class</h3>
                  <p className="mock-scientific-name">Scientific Taxonomy Name</p>
                </div>

                {/* Confidence Bar Placeholder */}
                <div className="confidence-wrapper">
                  <div className="confidence-labels">
                    <span className="confidence-title">
                      <BarChart2 className="inline-icon" /> Model Confidence Score
                    </span>
                    <span className="confidence-score-val">--%</span>
                  </div>
                  <div className="confidence-track">
                    <div className="confidence-fill-placeholder"></div>
                  </div>
                </div>

                {/* Botanical Metadata Cards */}
                <div className="metadata-placeholder-grid">
                  <div className="meta-card">
                    <CheckCircle2 className="meta-icon" />
                    <div>
                      <strong>Leaf Status</strong>
                      <span>Health Check (Step 2)</span>
                    </div>
                  </div>
                  <div className="meta-card">
                    <BookOpen className="meta-icon" />
                    <div>
                      <strong>Family Class</strong>
                      <span>Botanical Taxon (Step 2)</span>
                    </div>
                  </div>
                </div>

                <div className="placeholder-viva-note">
                  <strong>Viva Explanation Note:</strong> In Step 2, the Flask endpoint <code>POST /api/predict</code> will receive this image, run model inference via TensorFlow/Keras MobileNetV2, and return JSON containing the top predicted species and confidence percentage.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
