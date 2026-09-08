import React, { useState, useRef } from 'react';
import { Camera, Image as ImageIcon, UploadCloud, Trash2, RefreshCw, AlertTriangle, CheckCircle, Sparkles, Check } from 'lucide-react';
import './LeafUploader.css';

// Step 9 File Constraints
const ALLOWED_FORMATS = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const MAX_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB

/**
 * LeafUploader Component (Step 9 Updated)
 * ---------------------------------------
 * Mobile-first leaf image uploader supporting:
 *   1. 📷 Take Photo (Camera capture="environment")
 *   2. 🖼️ Choose from Gallery (File picker for JPG/PNG/WEBP)
 *   3. 💻 Desktop drag-and-drop
 *   4. Image Quality Guidance ("Tips for better results")
 *   5. Connection error fallback when Flask backend is unreachable.
 */
export default function LeafUploader({
  selectedImage,
  setSelectedImage,
  onAnalyzeSuccess,
  isAnalyzing,
  setIsAnalyzing
}) {
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState('');
  
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);

  /**
   * Client-Side File Validation
   */
  const validateAndProcessFile = (file) => {
    setValidationError('');

    if (!file) return;

    // 1. Validate File Format (JPG, JPEG, PNG, WEBP)
    if (!ALLOWED_FORMATS.includes(file.type.toLowerCase()) && !file.type.startsWith('image/')) {
      setValidationError('Please upload a valid JPG, PNG, or WEBP image.');
      return;
    }

    // 2. Validate File Size (Max 5 MB)
    if (file.size > MAX_SIZE_BYTES) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      setValidationError(`Image size (${sizeMB} MB) exceeds the 5 MB limit.`);
      return;
    }

    // 3. Read image for preview state
    const reader = new FileReader();
    reader.onload = () => {
      setSelectedImage({
        file: file,
        name: file.name || 'Captured_Leaf_Photo.jpg',
        sizeFormatted: file.size < 1024 * 1024 
          ? (file.size / 1024).toFixed(1) + ' KB' 
          : (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        previewUrl: reader.result,
        type: file.type ? file.type.replace('image/', '').toUpperCase() : 'JPG'
      });
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
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
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setValidationError('');
    if (cameraInputRef.current) cameraInputRef.current.value = '';
    if (galleryInputRef.current) galleryInputRef.current.value = '';
  };

  /**
   * Posts FormData under key 'image' to Flask POST /api/predict for real AI inference
   */
  const handleAnalyzeClick = async () => {
    if (!selectedImage || !selectedImage.file || isAnalyzing) return;

    setIsAnalyzing(true);
    setValidationError('');

    try {
      const formData = new FormData();
      formData.append('image', selectedImage.file);

      const apiBase = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiBase}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok && data.status === 'success') {
        onAnalyzeSuccess(data);
      } else {
        setValidationError(data.message || 'Prediction failed. Please try another leaf image.');
      }
    } catch (err) {
      setValidationError("You're offline or the AI server is unavailable. Please reconnect to analyze a leaf.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section id="upload" className="section">
      <div className="section-header-box">
        <h2 className="section-title">Identify Your Leaf</h2>
        <p className="section-subtitle">
          Take a clear photo with your camera or select an existing leaf photograph (JPG, PNG, WEBP — Max 5 MB).
        </p>
      </div>

      {/* Validation / Connection Error Alert Banner */}
      {validationError && (
        <div className="validation-error-alert" role="alert" aria-live="assertive">
          <AlertTriangle className="alert-icon" />
          <div className="alert-text">{validationError}</div>
        </div>
      )}

      <div className="uploader-card glass-card">
        {/* Hidden Camera Input (Mobile capture="environment") */}
        <input
          type="file"
          ref={cameraInputRef}
          onChange={handleFileChange}
          accept="image/*"
          capture="environment"
          style={{ display: 'none' }}
          aria-label="Take photo with mobile camera"
        />

        {/* Hidden Gallery / File Picker Input */}
        <input
          type="file"
          ref={galleryInputRef}
          onChange={handleFileChange}
          accept="image/jpeg,image/jpg,image/png,image/webp"
          style={{ display: 'none' }}
          aria-label="Choose leaf image from gallery"
        />

        {!selectedImage ? (
          /* State 1: Selection Dropzone & Dual Camera/Gallery Buttons */
          <div
            className={`dropzone-area ${dragActive ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-icon-circle">
              <UploadCloud className="upload-cloud-icon" />
            </div>

            <h3 className="dropzone-heading">Take a photo or upload leaf image</h3>
            <p className="dropzone-subheading">Select your preferred upload method below</p>

            {/* Mobile Dual Action Buttons: Camera & Gallery */}
            <div className="upload-input-options">
              <button 
                type="button" 
                className="btn btn-primary input-option-btn camera-btn"
                onClick={() => cameraInputRef.current?.click()}
                aria-label="Take leaf photo using camera"
              >
                <Camera className="option-btn-icon" />
                <span>Take Photo</span>
              </button>

              <button 
                type="button" 
                className="btn btn-outline input-option-btn gallery-btn"
                onClick={() => galleryInputRef.current?.click()}
                aria-label="Choose leaf image from photo gallery"
              >
                <ImageIcon className="option-btn-icon" />
                <span>Choose from Gallery</span>
              </button>
            </div>

            <p className="dropzone-specs">
              Supported Formats: <strong>JPG, PNG, WEBP</strong> (Max file size: <strong>5 MB</strong>)
            </p>
          </div>
        ) : (
          /* State 2: Selected Image Preview Mode */
          <div className="preview-layout">
            <div className="preview-frame">
              <img
                src={selectedImage.previewUrl}
                alt={`Preview of selected leaf image: ${selectedImage.name}`}
                className="preview-image"
              />
            </div>

            <div className="preview-info-box">
              <div className="preview-header">
                <ImageIcon className="info-file-icon" />
                <div className="file-text-meta">
                  <h4 className="preview-filename">{selectedImage.name}</h4>
                  <div className="file-pills">
                    <span className="file-pill">{selectedImage.sizeFormatted}</span>
                    <span className="file-pill format-pill">{selectedImage.type}</span>
                  </div>
                </div>
              </div>

              <div className="validation-success-badge">
                <CheckCircle className="badge-check-icon" />
                <span>Image validated (&lt; 5 MB)</span>
              </div>

              {/* Action Buttons */}
              <div className="uploader-actions">
                <button
                  type="button"
                  className="btn btn-outline action-btn"
                  onClick={() => galleryInputRef.current?.click()}
                  disabled={isAnalyzing}
                >
                  <RefreshCw className="btn-action-icon" /> Change Image
                </button>

                <button
                  type="button"
                  className="btn btn-secondary action-btn remove-btn"
                  onClick={handleRemoveImage}
                  disabled={isAnalyzing}
                >
                  <Trash2 className="btn-action-icon" /> Remove
                </button>

                <button
                  type="button"
                  className="btn btn-primary action-btn analyze-btn"
                  onClick={handleAnalyzeClick}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="btn-action-icon spin" /> Analyzing leaf...
                    </>
                  ) : (
                    <>
                      <Sparkles className="btn-action-icon" /> Analyze Leaf
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Image Quality Guidance Section ("Tips for better results") */}
      <div className="quality-tips-card glass-card">
        <h4 className="tips-card-heading">Tips for Better Identification Results</h4>
        <ul className="tips-list">
          <li className="tip-item">
            <Check className="tip-check-icon" />
            <span>Use a clear, focused photograph of a single leaf</span>
          </li>
          <li className="tip-item">
            <Check className="tip-check-icon" />
            <span>Ensure good lighting without extreme shadows</span>
          </li>
          <li className="tip-item">
            <Check className="tip-check-icon" />
            <span>Keep most of the leaf surface visible</span>
          </li>
          <li className="tip-item">
            <Check className="tip-check-icon" />
            <span>Place against a plain or high-contrast background if possible</span>
          </li>
        </ul>
      </div>
    </section>
  );
}
