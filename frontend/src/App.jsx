import React, { useState } from "react";
import './App.css'

function App() {
  const [userImage, setUserImage] = useState(null);
  const [uploadMode, setUploadMode] = useState('single'); // 'single' or 'multiple'
  const [garments, setGarments] = useState([]); // array to store multiple or single garments
  const [resultImage, setResultImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUserImageUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      setUserImage(URL.createObjectURL(e.target.files[0]));
      setResultImage(null);
    }
  };

  const handleGarmentImageUpload = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files).map(file => URL.createObjectURL(file));
      if (uploadMode === 'single') {
        setGarments([newFiles[0]]);
      } else {
        setGarments(prev => [...prev, ...newFiles]);
      }
      setResultImage(null);
    }
  };

  const removeGarment = (indexToRemove) => {
    setGarments(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const handleTryOn = () => {
    if (!userImage || garments.length === 0) {
      alert("Please upload both a user image and at least one garment.");
      return;
    }
    
    setIsProcessing(true);
    // Simulate ML processing delay
    setTimeout(() => {
      // Mock result (in real application, backend provides image URL)
      setResultImage(userImage); 
      setIsProcessing(false);
    }, 4000);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>STYLE YOUR FIT</h1>
        <p className="subtitle">Personalized Virtual Clothing Try-On System</p>
      </header>

      <main className="main-content">
        <section className="input-section">
          <h2>1. Input Photos</h2>
          
          <div className="upload-container">
            <div className="upload-box card user-box">
              <h3>Person Image</h3>
              {userImage ? (
                <div className="image-preview">
                  <img src={userImage} alt="User" />
                  <button onClick={() => setUserImage(null)} className="btn-secondary">Remove</button>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <label htmlFor="user-upload" className="btn-primary upload-label">
                    Upload Body Image
                  </label>
                  <input id="user-upload" type="file" accept="image/*" onChange={handleUserImageUpload} hidden />
                  <p className="sub-text">Full-body front view</p>
                </div>
              )}
            </div>

            <div className="upload-box card garment-box">
              <h3>Garment Images</h3>
              
              <div className="mode-toggle">
                <button 
                  className={`toggle-btn ${uploadMode === 'single' ? 'active' : ''}`}
                  onClick={() => { setUploadMode('single'); setGarments([]); }}
                >
                  Single Garment
                </button>
                <button 
                  className={`toggle-btn ${uploadMode === 'multiple' ? 'active' : ''}`}
                  onClick={() => { setUploadMode('multiple'); setGarments([]); }}
                >
                  Multiple Garments
                </button>
              </div>

              <div className="garment-list">
                {garments.length > 0 && garments.map((g, index) => (
                  <div key={index} className="image-preview item-preview">
                    <img src={g} alt={`Garment ${index + 1}`} />
                    <button onClick={() => removeGarment(index)} className="btn-secondary btn-sm">Remove</button>
                  </div>
                ))}
              </div>

              {(uploadMode === 'multiple' || garments.length === 0) && (
                <div className="upload-placeholder mt-3">
                  <label htmlFor="garment-upload" className="btn-primary upload-label">
                    Upload Garment{uploadMode === 'multiple' ? 's' : ''}
                  </label>
                  <input 
                    id="garment-upload" 
                    type="file" 
                    accept="image/*" 
                    multiple={uploadMode === 'multiple'} 
                    onChange={handleGarmentImageUpload} 
                    hidden 
                  />
                  <p className="sub-text">Clothes flat lay or plain background</p>
                </div>
              )}
            </div>
          </div>

          <button 
            className={`btn-action ${(!userImage || garments.length === 0 || isProcessing) ? 'disabled' : ''}`} 
            onClick={handleTryOn} 
            disabled={!userImage || garments.length === 0 || isProcessing}
          >
            {isProcessing ? "Processing..." : "Generate Virtual Try-On"}
          </button>
        </section>

        <section className="output-section card">
          <h2>2. Try-On Result</h2>
          <div className="result-container">
            {isProcessing ? (
              <div className="loader">
                <div className="spinner"></div>
                <h4>Aligning & Segementing...</h4>
                <p className="ml-status">Body Detection -&gt; Pose Estimation -&gt; Garment Warping</p>
              </div>
            ) : resultImage ? (
              <div className="result-wrapper">
                <img src={resultImage} alt="Result Output" className="result-img" />
                <button className="btn-primary download-btn">Download</button>
              </div>
            ) : (
              <div className="result-placeholder">
                <p>Output image will appear here after processing.</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
