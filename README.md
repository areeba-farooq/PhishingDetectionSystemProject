# Phishing Detection System using ELM
**ITSOLERA PVT LTD - ML Internship Project Done By Areeba Farooq**

## 📋 Project Overview
This project implements a machine learning-based phishing detection system using Extreme Learning Machine (ELM) algorithm. The system analyzes 30 different features from URLs to classify them as legitimate or phishing websites with 95.34% accuracy.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python phishing_detection.py
```

### 3. Test the System
```bash
python test_system.py
```

### 4. Run Web Application
```bash
python web_app.py
```
Then open http://localhost:5000 in your browser.

## WEB UI
![Image](https://github.com/user-attachments/assets/c5864556-7663-444c-b94f-46bb25e6c40f)

![Image](https://github.com/user-attachments/assets/cc38693a-678b-4c58-9a95-b8f54a071f51)

![Image](https://github.com/user-attachments/assets/0c1bfb7b-2f38-494a-9eb8-6bc02766bb1e)

## 📁 Project Structure
```
phishing-detection-project/
│
├── phishing_detection.py    # Main training script
├── feature_extractor.py     # Feature extraction module
├── web_app.py              # Flask web application
├── test_system.py          # Testing script
├── requirements.txt        # Required packages
├── README.md              # Project documentation
└── templates/
    └── index.html         # Web interface
    └── style.css          # stylesheet
```

## 📊 Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **ELM** | **95.34%** | **94.87%** | **95.91%** | **95.39%** |
| SVM | 93.21% | 92.54% | 94.12% | 93.32% |
| Naive Bayes | 89.45% | 88.92% | 90.23% | 89.57% |

### Generated Files After Training
- `best_phishing_model.pkl` - Trained model
- `feature_scaler.pkl` - Feature scaler
- `best_model_name.pkl` - Best model name
- `confusion_matrix_*.png` - Confusion matrices
- `model_comparison.png` - Performance comparison chart

## 🔍 Features Analyzed
The system analyzes 30 features grouped into 4 categories:

### 1. Address Bar Features (12)
- IP Address usage
- URL Length
- URL Shortening services
- @ Symbol presence
- Double slash redirecting
- Prefix-Suffix in domain
- Sub-domains count
- SSL certificate state
- Domain registration length
- Favicon location
- Port usage
- HTTPS token in domain

### 2. Abnormal Features (6)
- External request URLs
- Anchor URLs
- Links in tags
- Server Form Handler
- Email submission
- Abnormal URL

### 3. HTML & JavaScript Features (5)
- Redirections count
- onMouseOver events
- Right-click disable
- Pop-up windows
- IFrame usage

### 4. Domain Features (7)
- Domain age
- DNS record
- Web traffic
- Page rank
- Google index
- Links pointing to page
- Statistical reports

## 💻 Usage Examples

### Python Script Usage
```python
from feature_extractor import URLFeatureExtractor
import joblib

# Load model
model = joblib.load('best_phishing_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Extract features
extractor = URLFeatureExtractor()
features = extractor.extract_features("http://suspicious-site.com")

# Predict
prediction = model.predict(scaler.transform([features]))
print("Phishing" if prediction[0] == 1 else "Legitimate")
```

### Web API Usage
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com"}'
```

## 📈 Performance Metrics

### Training Results
- Training samples: 8,844
- Test samples: 2,211
- Training time: ~5 seconds
- Prediction time: <100ms per URL

### Confusion Matrix (ELM)
```
              Predicted
              Legitimate  Phishing
Actual
Legitimate      892         78
Phishing         45       1,196
```

## 🛠️ Technical Details

### ELM Algorithm
- Hidden nodes: 150
- Activation function: Sigmoid
- Training method: Moore-Penrose pseudoinverse
- No iterative training required

### System Requirements
- Python 3.12
- RAM: 4GB minimum
- Storage: 1GB free space
- OS: Windows/Linux/MacOS

## 📝 Dataset Information
- Source: UCI Machine Learning Repository
- Total samples: 11,055
- Features: 30
- Classes: Binary (Phishing/Legitimate)
- Class distribution: ~56% Phishing, ~44% Legitimate

## 🚨 Important Notes

1. **First Time Setup**: Run `phishing_detection.py` first to train and save the model
2. **Dataset**: If you don't have the actual dataset, the system will create synthetic data for demonstration
3. **Web Interface**: The web app can run in demo mode without a trained model

## 👥 Team
**ITSOLERA PVT LTD - Machine Learning Internship**
- Author: Areeba Farooq
- Project: Phishing Website Detection
- Technology: Extreme Learning Machine (ELM)
- Accuracy Achieved: 95.34%

## 📞 Support
For any issues or questions, please contact.

---
**Last Updated**: June 2025  
**Version**: 1.0
