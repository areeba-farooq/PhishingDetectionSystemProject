import joblib
import numpy as np
from feature_extractor import URLFeatureExtractor

def test_prediction_system():
    """Test the complete prediction system"""
    print("="*60)
    print("TESTING PHISHING DETECTION SYSTEM")
    print("="*60)
    
    # Load model and scaler
    try:
        model = joblib.load('best_phishing_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        model_name = joblib.load('best_model_name.pkl')
        print(f"\n✓ Successfully loaded {model_name} model")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("Please run phishing_detection.py first to train the model.")
        return
    
    # Initialize feature extractor
    extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        # Legitimate URLs
        ("https://www.google.com", "LEGITIMATE"),
        ("https://www.facebook.com", "LEGITIMATE"),
        ("https://www.amazon.com", "LEGITIMATE"),
        ("https://www.microsoft.com", "LEGITIMATE"),
        
        # Suspicious/Phishing URLs
        ("http://192.168.1.1/secure-login", "PHISHING"),
        ("http://google.com@phishing-site.com", "PHISHING"),
        ("https://bit.ly/2suspicious", "PHISHING"),
        ("http://paypal-verification.fake-domain.com", "PHISHING"),
        ("http://www.google.com.fake.com", "PHISHING"),
        ("http://login-bank-secure.000webhostapp.com", "PHISHING"),
    ]
    
    print("\nTesting on sample URLs:")
    print("-"*80)
    print(f"{'URL':<50} {'Expected':<12} {'Predicted':<12} {'Status':<10}")
    print("-"*80)
    
    correct_predictions = 0
    
    for url, expected in test_urls:
        # Extract features
        features = extractor.extract_features(url)
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        predicted_label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        
        # Check if correct
        is_correct = predicted_label == expected
        if is_correct:
            correct_predictions += 1
        
        status = "✓" if is_correct else "✗"
        
        # Truncate URL for display
        display_url = url[:47] + "..." if len(url) > 50 else url
        
        print(f"{display_url:<50} {expected:<12} {predicted_label:<12} {status:<10}")
    
    # Calculate accuracy
    accuracy = correct_predictions / len(test_urls) * 100
    
    print("-"*80)
    print(f"\nTest Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_urls)} correct)")
    
    # Feature importance test
    print("\n\nFeature Analysis for a phishing URL:")
    print("-"*50)
    
    phishing_url = "http://192.168.1.1/paypal-login"
    features = extractor.extract_features(phishing_url)
    
    feature_names = [
        "IP Address", "URL Length", "Shortening Service", "@ Symbol",
        "Double Slash", "Prefix-Suffix", "Sub Domains", "SSL State"
    ]
    
    print(f"URL: {phishing_url}")
    print("\nKey Features:")
    for i, (name, value) in enumerate(zip(feature_names, features[:8])):
        interpretation = "Suspicious" if value == 1 else "Normal" if value == -1 else "Neutral"
        print(f"  {name:<20}: {interpretation:<12} (value: {value})")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED!")
    print("="*60)

def test_bulk_prediction():
    """Test bulk URL prediction"""
    print("\n\nBULK PREDICTION TEST")
    print("-"*50)
    
    # Simulate bulk URLs
    bulk_urls = [
        "https://www.linkedin.com",
        "http://suspicious-site.tk",
        "https://www.github.com",
        "http://192.168.0.1/admin",
        "https://bit.ly/win-prize",
        "https://www.youtube.com",
        "http://paypal-update.fake.com",
        "https://www.stackoverflow.com"
    ]
    
    try:
        model = joblib.load('best_phishing_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        extractor = URLFeatureExtractor()
        
        phishing_count = 0
        legitimate_count = 0
        
        print(f"Processing {len(bulk_urls)} URLs...\n")
        
        for url in bulk_urls:
            features = extractor.extract_features(url)
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            
            if prediction == 1:
                phishing_count += 1
                print(f"⚠️  {url} - PHISHING")
            else:
                legitimate_count += 1
                print(f"✓  {url} - LEGITIMATE")
        
        print(f"\nSummary:")
        print(f"Total URLs: {len(bulk_urls)}")
        print(f"Phishing: {phishing_count} ({phishing_count/len(bulk_urls)*100:.1f}%)")
        print(f"Legitimate: {legitimate_count} ({legitimate_count/len(bulk_urls)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error in bulk prediction: {e}")

if __name__ == "__main__":
    # Run all tests
    test_prediction_system()
    test_bulk_prediction()