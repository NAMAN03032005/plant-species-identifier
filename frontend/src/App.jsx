import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import LeafUploader from './components/LeafUploader';
import PredictionResult, { parseClassLabel } from './components/PredictionResult';
import HowItWorks from './components/HowItWorks';
import AboutSection from './components/AboutSection';
import Footer from './components/Footer';
import InstallPrompt from './components/InstallPrompt';

/**
 * Main App Component (LeafScan Step 9 - Mobile-First PWA)
 * --------------------------------------------------------
 * Manages full-stack real AI image predictions, PWA install prompts,
 * camera/gallery inputs, and session-based recent analysis history.
 */
export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Session-based history (keeps last 3 real predictions in React memory)
  const [recentAnalyses, setRecentAnalyses] = useState([]);

  // Reset handler when user removes/changes image or clicks "Analyze Another Leaf"
  const handleReset = () => {
    setSelectedImage(null);
    setPredictionData(null);
    const uploadElem = document.getElementById('upload');
    if (uploadElem) {
      uploadElem.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleImageSelectionChange = (imageObj) => {
    setSelectedImage(imageObj);
    setPredictionData(null);
  };

  const handleAnalyzeSuccess = (serverResponse) => {
    setPredictionData(serverResponse);
    
    // Save to session history (max 3 items)
    if (serverResponse?.prediction) {
      const parsed = parseClassLabel(serverResponse.prediction.species);
      const historyItem = {
        id: Date.now(),
        plant: parsed.plant,
        condition: parsed.condition,
        confidencePct: (serverResponse.prediction.confidence * 100).toFixed(2),
        previewUrl: selectedImage?.previewUrl
      };
      setRecentAnalyses(prev => [historyItem, ...prev.slice(0, 2)]);
    }

    // Smooth scroll to prediction result section
    const resultsElem = document.getElementById('results');
    if (resultsElem) {
      resultsElem.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="app-layout">
      <Navbar />
      <main>
        <Hero />
        <LeafUploader 
          selectedImage={selectedImage}
          setSelectedImage={handleImageSelectionChange}
          onAnalyzeSuccess={handleAnalyzeSuccess}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
        />
        <PredictionResult 
          selectedImage={selectedImage}
          predictionData={predictionData}
          onReset={handleReset}
        />
        <HowItWorks />
        <AboutSection />
      </main>
      <Footer />
      <InstallPrompt />
    </div>
  );
}
