import React from 'react';
import { Cpu, CheckCircle2, AlertCircle, BarChart2, Leaf, Sparkles, Award, RotateCcw, Info, Cpu as CpuIcon, Layers, ShieldCheck } from 'lucide-react';
import './PredictionResult.css';

/**
 * Parse raw model class name into Plant Species and Leaf Condition.
 * e.g., 'Apple___healthy' -> { plant: 'Apple', condition: 'Healthy', isHealthy: true }
 * e.g., 'Apple___Black_rot' -> { plant: 'Apple', condition: 'Black Rot', isHealthy: false }
 */
export function parseClassLabel(rawName) {
  if (!rawName) return { plant: 'Unknown', condition: 'Unknown', isHealthy: true, raw: '' };
  
  const parts = rawName.split('___');
  if (parts.length === 2) {
    let plant = parts[0].replace(/_/g, ' ');
    if (plant.toLowerCase() === 'corn') plant = 'Corn (Maize)';

    let condition = parts[1].replace(/_/g, ' ');
    condition = condition
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');

    const isHealthy = condition.toLowerCase() === 'healthy';
    return { plant, condition, isHealthy, raw: rawName };
  }

  return { plant: rawName.replace(/_/g, ' '), condition: 'Identified', isHealthy: true, raw: rawName };
}

/**
 * PredictionResult Component (Step 6 Polished)
 * --------------------------------------------
 * Displays real AI analysis results parsed by Plant Species and Leaf Condition,
 * ARIA accessible confidence meter, top-3 candidates, 'About the AI' metadata,
 * academic limitation disclaimer, and 'Analyze Another Leaf' reset workflow.
 */
export default function PredictionResult({ selectedImage, predictionData, onReset }) {
  const top1Raw = predictionData?.prediction;
  const top3Raw = predictionData?.top_predictions || [];

  const top1Parsed = top1Raw ? parseClassLabel(top1Raw.species) : null;
  const top1ConfidencePct = top1Raw ? (top1Raw.confidence * 100).toFixed(2) : '0.00';

  return (
    <section id="results" className="section">
      <h2 className="section-title">Analysis Result</h2>
      <p className="section-subtitle">
        Real-time MobileNetV2 Deep Learning output for Plant Species & Leaf Condition identification.
      </p>

      <div className="results-card glass-card">
        {!selectedImage ? (
          /* State 1: Initial Empty State */
          <div className="empty-results-view">
            <div className="empty-icon-wrapper">
              <Cpu className="empty-cpu-icon" />
            </div>
            <h3 className="empty-heading">Your prediction will appear here.</h3>
            <p className="empty-subtext">
              Select a plant leaf image in the section above and click "Analyze Leaf" to begin.
            </p>
          </div>
        ) : !predictionData ? (
          /* State 2: Image Selected, Awaiting "Analyze Leaf" Click */
          <div className="ready-to-analyze-view">
            <div className="ready-thumb-box">
              <img src={selectedImage.previewUrl} alt="Selected leaf" className="ready-thumb" />
            </div>
            <div className="ready-text-box">
              <div className="status-tag">Image Ready</div>
              <h3 className="ready-heading">Ready for AI Analysis</h3>
              <p className="ready-desc">
                Click <strong>"Analyze Leaf"</strong> above to evaluate this photograph with MobileNetV2.
              </p>
            </div>
          </div>
        ) : (
          /* State 3: Polished Real AI Prediction Output */
          <div className="prediction-active-view">
            {/* Header Badge */}
            <div className="prediction-header-bar">
              <div className="ai-status-pill">
                <Sparkles className="pill-sparkle-icon" />
                <span>Plant & Leaf Condition Identification</span>
              </div>
              <span className="raw-class-badge" title="Raw Keras Class Label">
                Class: {top1Raw.species}
              </span>
            </div>

            {/* Main Image + Result Grid */}
            <div className="prediction-main-grid">
              {/* Left Column: Image Thumbnail */}
              <div className="prediction-thumb-card">
                <img src={selectedImage.previewUrl} alt="Analyzed leaf photograph" className="prediction-leaf-img" />
                <span className="thumb-caption">Uploaded Leaf Image</span>
                
                {/* Reset / Analyze Another Button */}
                <button
                  type="button"
                  className="btn btn-secondary analyze-another-btn"
                  onClick={onReset}
                >
                  <RotateCcw className="btn-icon-sm" /> Analyze Another Leaf
                </button>
              </div>

              {/* Right Column: Parsed Taxonomy & Confidence */}
              <div className="prediction-details-card">
                
                {/* Plant Species & Leaf Condition Cards */}
                <div className="taxonomy-grid">
                  <div className="taxonomy-card">
                    <div className="tax-label">
                      <Leaf className="tax-icon" /> Plant Species
                    </div>
                    <div className="tax-value">{top1Parsed.plant}</div>
                  </div>

                  <div className={`taxonomy-card ${top1Parsed.isHealthy ? 'healthy-card' : 'disease-card'}`}>
                    <div className="tax-label">
                      {top1Parsed.isHealthy ? (
                        <CheckCircle2 className="tax-icon green-icon" />
                      ) : (
                        <AlertCircle className="tax-icon red-icon" />
                      )}
                      Leaf Condition
                    </div>
                    <div className="tax-value">{top1Parsed.condition}</div>
                  </div>
                </div>

                {/* Accessible Confidence Score Bar */}
                <div className="confidence-meter-container">
                  <div className="confidence-label-row">
                    <span className="confidence-text">
                      <Award className="award-icon" /> Model Confidence Score
                    </span>
                    <span className="confidence-percentage">{top1ConfidencePct}%</span>
                  </div>
                  
                  {/* ARIA Accessible Progress Bar */}
                  <div
                    className="confidence-track-bar"
                    role="progressbar"
                    aria-valuenow={parseFloat(top1ConfidencePct)}
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-label={`Model prediction confidence score: ${top1ConfidencePct}%`}
                  >
                    <div
                      className="confidence-fill-bar"
                      style={{ width: `${Math.max(5, parseFloat(top1ConfidencePct))}%` }}
                    ></div>
                  </div>

                  <p className="confidence-disclaimer">
                    <Info className="info-inline-icon" /> Confidence represents the model's probability for this prediction. It does not guarantee that the prediction is correct.
                  </p>
                </div>

                {/* Top 3 Predictions List */}
                <div className="top3-predictions-box">
                  <h4 className="top3-title">
                    <BarChart2 className="chart-icon" /> Top 3 Model Predictions
                  </h4>
                  <ul className="top3-list">
                    {top3Raw.map((cand, idx) => {
                      const parsed = parseClassLabel(cand.species);
                      const pct = (cand.confidence * 100).toFixed(2);
                      return (
                        <li key={idx} className={`top3-item ${idx === 0 ? 'rank-1' : ''}`}>
                          <span className="rank-num">{idx + 1}.</span>
                          <div className="cand-text">
                            <strong>{parsed.plant}</strong> — <span>{parsed.condition}</span>
                          </div>
                          <div className="cand-bar-wrapper">
                            <div className="cand-mini-bar" style={{ width: `${Math.max(4, pct)}%` }}></div>
                          </div>
                          <span className="cand-pct">{pct}%</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>

              </div>
            </div>

            {/* About the AI & Academic Limitation Section */}
            <div className="ai-metadata-footer">
              <div className="about-ai-card">
                <h4><CpuIcon className="ai-meta-icon" /> About the AI Model</h4>
                <div className="ai-tags">
                  <span className="ai-tag"><Layers className="tag-icon" /> MobileNetV2</span>
                  <span className="ai-tag"><Sparkles className="tag-icon" /> Transfer Learning</span>
                  <span className="ai-tag"><ShieldCheck className="tag-icon" /> Pretrained ImageNet</span>
                  <span className="ai-tag">Input: 224 × 224 RGB</span>
                </div>
                <p className="ai-explanation">
                  LeafScan uses MobileNetV2 with transfer learning. The pretrained network extracts visual features 
                  from the leaf image, while the final classification layer predicts one of the 10 plant/leaf classes in the training dataset.
                </p>
              </div>

              <div className="limitation-notice-box">
                <Info className="limit-icon" />
                <p>
                  <strong>Academic Limitation Notice:</strong> LeafScan can only classify categories represented in its training dataset. 
                  Results may be affected by lighting, image quality, leaf orientation, background, and visual similarity between classes. 
                  <em>For educational and demonstration purposes only.</em>
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
