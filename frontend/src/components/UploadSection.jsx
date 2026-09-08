import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Trash2, Info, ArrowUpRight } from 'lucide-react';
import './UploadSection.css';

/**
 * UploadSection Component
 * -----------------------
 * Handles client-side leaf image selection and preview for Step 1.
 * Features drag-and-drop support and explicitly indicates that model
 * prediction integration takes place in Step 2.
 */
export default function UploadSection({ selectedImage, setSelectedImage }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    const reader = new FileReader();
    reader.onload = () => {
      setSelectedImage({
        file: file,
        name: file.name,
        size: (file.size / 1024).toFixed(1) + ' KB',
        previewUrl: reader.result,
      });
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <section id="upload" className="section">
      <h2 className="section-title">Upload Leaf Image</h2>
      <p className="section-subtitle">
        Select or drag a clear photograph of a plant leaf to prepare for deep learning identification.
      </p>

      {/* Step 1 Informational Banner */}
      <div className="step1-notice-banner">
        <Info className="notice-icon" />
        <div className="notice-content">
          <strong>Step 1 UI Demo Mode:</strong> Image preview and layout ready. Deep Learning prediction pipeline (MobileNetV2 model backend) will be connected in Step 2.
        </div>
      </div>

      <div className="upload-container glass-card">
        {!selectedImage ? (
          /* Dropzone state */
          <div
            className={`dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <div className="dropzone-icon-wrapper">
              <UploadCloud className="dropzone-icon" />
            </div>
            <h3 className="dropzone-title">Click to upload or drag & drop</h3>
            <p className="dropzone-hint">Supports PNG, JPG, JPEG, WEBP (Max 10MB)</p>
            <button className="btn btn-primary dropzone-btn" type="button">
              Select Image File
            </button>
          </div>
        ) : (
          /* Image Selected / Preview State */
          <div className="preview-card">
            <div className="preview-media">
              <img src={selectedImage.previewUrl} alt="Selected leaf preview" className="preview-img" />
            </div>
            <div className="preview-details">
              <div className="file-info">
                <ImageIcon className="file-icon" />
                <div>
                  <h4 className="file-name">{selectedImage.name}</h4>
                  <span className="file-size">{selectedImage.size}</span>
                </div>
              </div>

              <div className="preview-status-pill">
                <span className="pulse-dot"></span>
                <span>Image loaded into browser state</span>
              </div>

              <div className="preview-actions">
                <button 
                  className="btn btn-outline btn-clear" 
                  onClick={handleClear}
                >
                  <Trash2 className="action-icon" /> Remove Image
                </button>
                
                <a href="#results" className="btn btn-primary btn-predict-disabled">
                  Ready for Model (Step 2) <ArrowUpRight className="action-icon" />
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
