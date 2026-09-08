import React from 'react';
import { Upload, Cpu, Award, Sparkles } from 'lucide-react';
import './HowItWorks.css';

/**
 * HowItWorks Component
 * -------------------
 * Visualizes the 3-step deep learning workflow: Upload -> Analyze -> Identify.
 */
export default function HowItWorks() {
  const steps = [
    {
      number: '01',
      title: 'Upload',
      description: 'User uploads a clean photograph of a plant leaf.',
      icon: Upload,
      detail: 'Supports PNG, JPG, or WEBP image formats.'
    },
    {
      number: '02',
      title: 'Analyze',
      description: 'The image is processed by a deep learning model based on transfer learning.',
      icon: Cpu,
      detail: 'MobileNetV2 extracts deep visual feature maps (Step 2 Integration).'
    },
    {
      number: '03',
      title: 'Identify',
      description: 'The application displays the predicted plant species and confidence score.',
      icon: Award,
      detail: 'Softmax activation layer outputs top species classification.'
    }
  ];

  return (
    <section id="how-it-works" className="section">
      <h2 className="section-title">How It Works</h2>
      <p className="section-subtitle">
        A streamlined 3-step pipeline converting visual leaf inputs into precise botanical classifications.
      </p>

      <div className="how-it-works-grid">
        {steps.map((step, idx) => {
          const IconComponent = step.icon;
          return (
            <div key={idx} className="step-card glass-card">
              <div className="step-number">{step.number}</div>
              <div className="step-icon-box">
                <IconComponent className="step-icon" />
              </div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-description">{step.description}</p>
              <div className="step-detail-pill">{step.detail}</div>
            </div>
          );
        })}
      </div>

      <div className="pipeline-notice">
        <Sparkles className="notice-sparkle" />
        <span>
          <strong>Pipeline Status:</strong> Step 1 UI & API interface ready. Step 2 will wire up the live TensorFlow model inference pipeline.
        </span>
      </div>
    </section>
  );
}
