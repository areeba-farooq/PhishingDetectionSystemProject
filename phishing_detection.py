import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExtremeLearningMachine:
    """ELM Classifier Implementation"""
    def __init__(self, n_hidden=100, activation='sigmoid', random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _activation_func(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Random weights between input and hidden layer
        self.input_weights = np.random.randn(n_features, self.n_hidden)
        self.biases = np.random.randn(self.n_hidden)
        
        # Calculate hidden layer output
        H = self._activation_func(X.dot(self.input_weights) + self.biases)
        
        # Calculate output weights using pseudoinverse
        self.output_weights = np.linalg.pinv(H).dot(y)
        
        return self
    
    def predict(self, X):
        H = self._activation_func(X.dot(self.input_weights) + self.biases)
        output = H.dot(self.output_weights)
        return (output > 0.5).astype(int)

def load_arff_dataset(filename='Training_Dataset.arff'):
    """Load dataset from ARFF file"""
    print(f"Loading dataset from {filename}...")
    
    # Column names based on the documentation
    columns = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWindow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report', 'Result'
    ]
    
    try:
        # Method 1: Try using scipy to read ARFF
        try:
            from scipy.io import arff
            data, meta = arff.loadarff(filename)
            df = pd.DataFrame(data)
            
            # Convert byte strings to integers if needed
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.decode('utf-8').astype(int)
                    
        except ImportError:
            print("scipy not installed. Reading ARFF file manually...")
            
            # Method 2: Manual ARFF parsing
            data_started = False
            data_lines = []
            
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('%'):
                        continue
                    
                    # Check if data section started
                    if line.upper().startswith('@DATA'):
                        data_started = True
                        continue
                    
                    # Read data lines
                    if data_started and line:
                        # Remove quotes and split by comma
                        values = line.replace("'", "").split(',')
                        data_lines.append(values)
            
            # Convert to DataFrame
            df = pd.DataFrame(data_lines)
            
            # Convert to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Assign column names
            if len(df.columns) == len(columns):
                df.columns = columns
            else:
                print(f"Warning: Expected {len(columns)} columns but found {len(df.columns)}")
                # Use generic column names
                df.columns = [f'feature_{i}' for i in range(len(df.columns)-1)] + ['Result']
        
        print(f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]-1} features")
        
        # Check data statistics
        print("\nData Statistics:")
        print(f"Unique values in Result column: {df['Result'].unique()}")
        print(f"Value counts:\n{df['Result'].value_counts()}")
        
        return df
        
    except FileNotFoundError:
        print(f"\nError: File '{filename}' not found!")
        print("Please make sure 'Training_Dataset.arff' is in the same directory as this script.")
        print("\nCreating synthetic data for demonstration...")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 11055
        
        data = []
        for i in range(n_samples):
            features = []
            is_phishing = np.random.random() < 0.56
            
            for j in range(30):
                if is_phishing:
                    if j in [0, 3, 4, 16]:
                        features.append(1 if np.random.random() < 0.7 else -1)
                    elif j in [7, 24]:
                        features.append(-1 if np.random.random() < 0.7 else 1)
                    else:
                        features.append(np.random.choice([-1, 0, 1], p=[0.2, 0.3, 0.5]))
                else:
                    if j in [0, 3, 4, 16]:
                        features.append(-1 if np.random.random() < 0.8 else 1)
                    elif j in [7, 24]:
                        features.append(1 if np.random.random() < 0.8 else -1)
                    else:
                        features.append(np.random.choice([-1, 0, 1], p=[0.5, 0.3, 0.2]))
            
            features.append(1 if is_phishing else -1)
            data.append(features)
        
        df = pd.DataFrame(data, columns=columns)
        return df

def preprocess_data(df):
    """Preprocess the dataset"""
    result_values = df['Result'].unique()
    print(f"\nUnique Result values: {result_values}")
    
    # Convert Result to binary (0 for legitimate, 1 for phishing)
    if -1 in result_values and 1 in result_values:
        # Original format: -1 for legitimate, 1 for phishing
        df['Result'] = df['Result'].apply(lambda x: 1 if x == 1 else 0)
    elif 0 in result_values and 1 in result_values:
        pass
    else:
        print(f"Unexpected Result values: {result_values}")
        # Assume first unique value is legitimate, second is phishing
        df['Result'] = df['Result'].apply(lambda x: 1 if x == result_values[1] else 0)
    
    df = df.dropna()
    
    X = df.drop('Result', axis=1).values
    y = df['Result'].values
    
    print(f"\nClass distribution after preprocessing:")
    print(f"Legitimate (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Phishing (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train all three models"""
    models = {}
    
    # 1. ELM
    print("\nTraining ELM...")
    models['ELM'] = ExtremeLearningMachine(n_hidden=150, activation='sigmoid')
    models['ELM'].fit(X_train, y_train)
    
    # 2. SVM
    print("Training SVM...")
    models['SVM'] = SVC(kernel='rbf', C=1.0, random_state=42)
    models['SVM'].fit(X_train, y_train)
    
    # 3. Naive Bayes
    print("Training Naive Bayes...")
    models['Naive Bayes'] = GaussianNB()
    models['Naive Bayes'].fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return results"""
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        print(f"Confusion matrix saved as: confusion_matrix_{name.lower().replace(' ', '_')}.png")
    
    return results

def plot_comparison(results):
    """Create comparison charts"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    print("\nModel comparison chart saved as: model_comparison.png")

def save_best_model(models, results, scaler):
    """Save the best performing model"""
    # Find best model based on accuracy
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    joblib.dump(best_model, 'best_phishing_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(best_model_name, 'best_model_name.pkl')
    
    print("Model files saved:")
    print("- best_phishing_model.pkl")
    print("- feature_scaler.pkl")
    print("- best_model_name.pkl")

def main():
    """Main execution function"""
    print("="*70)
    print("PHISHING DETECTION SYSTEM - ITSOLERA PVT LTD")
    print("="*70)
    
    # Load dataset from ARFF file
    df = load_arff_dataset('Training_Dataset.arff')
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    
    plot_comparison(results)
    
    save_best_model(models, results, scaler)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\nFINAL RESULTS SUMMARY:")
    print("-"*50)
    for model_name in results:
        print(f"{model_name:15} - Accuracy: {results[model_name]['accuracy']:.4f}")
    print("-"*50)

if __name__ == "__main__":
    main()