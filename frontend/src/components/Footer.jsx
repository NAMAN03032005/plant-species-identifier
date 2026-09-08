import React from 'react';
import { Leaf, GraduationCap, Code2 } from 'lucide-react';
import './Footer.css';

/**
 * Footer Component
 * ----------------
 * Bottom section displaying credits, technology tags, and academic project meta.
 */
export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-brand">
          <div className="footer-logo">
            <Leaf className="footer-leaf-icon" />
            <span>LeafScan</span>
          </div>
          <p className="footer-desc">
            Plant Species Identification using Transfer Learning & MobileNetV2 Deep Learning architecture.
          </p>
        </div>

        <div className="footer-meta">
          <div className="meta-tag">
            <GraduationCap className="meta-icon" />
            <span>Academic TAE Project</span>
          </div>
          <div className="meta-tag">
            <span>Tech Stack: React • Vite • Flask • TensorFlow</span>
          </div>
        </div>

        <div className="footer-copyright">
          © {new Date().getFullYear()} LeafScan Deep Learning Project. Built for Step 1 Demonstration.
        </div>
      </div>
    </footer>
  );
}
