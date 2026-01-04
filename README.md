# Unified Multi-Channel Phishing Detection System

> **A Machine Learning framework for detecting social engineering attacks across Emails, SMS, and Malicious URLs using a unified Random Forest classifier.**

This project addresses the fragmentation in modern cybersecurity defenses by providing a single, unified machine learning pipeline. It consolidates data from three distinct vectors into a single pipeline, utilizing **Natural Language Processing (NLP)** to identify "phishing signals" (urgency, coercion, financial pressure) regardless of the medium.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-Educational-green)
- **Unified Protection:** Detects threats in **URLs**, **SMS** ("Smishing"), and **Emails** using a single model.
- **Signal-Preserving NLP:** Uses a custom tokenizer that retains critical punctuation (e.g., `!`, `$`, `%`) often stripped by standard cleaners, preserving the "intent" of the attack.
- **Robust ETL Pipeline:** Automatically ingests, cleans, and balances heterogeneous datasets (handling different CSV headers like `v1/v2`, `Subject/Body`, etc.).
- **Interactive CLI:** A user-friendly command-line interface for training models and testing specific text inputs.### Prerequisites
Ensure you have **Python 3.8+** installed.

---

## 📂 Repository Structure

```text
unified-phishing-detection/
│                  
├── models/                 # (Created automatically) Stores the trained 'phishing_model.pkl'
│
├── src/
│   ├── data_loader.py      # Interactive ETL pipeline: Ingests, cleans, and balances data
│   ├── preprocessing.py    # Custom NLP logic: Regex cleaning & Signal Preservation
│   ├── features.py         # TF-IDF Vectorizer configuration (Top 5k features)
│   ├── train_model.py      # Random Forest Pipeline construction & Training loop
│   ├── predict.py          # Inference Engine for real-time predictions
│   └── __init__.py
│
├── main.py                 # CLI Entry Point (Train, Test, Predict)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

### Setup
1. **Clone the repository**
   ```bash
   git clone [https://github.com/YourUsername/unified-phishing-detection.git](https://github.com/YourUsername/unified-phishing-detection.git)
   cd unified-phishing-detection
pip install -r requirements.txt

---

### **Section: Environment Variables** (Rename this to "**Datasets**" in the editor)

*Note: In Readme.so, you can click on the pencil icon next to the section name to rename "Environment Variables" to "Datasets".*


To train the model, download the following datasets and place them in the `data/` folder (or upload them when prompted by the script).

| Channel | Dataset Name | Source Link | Instructions                                                                                                     |
| :--- | :--- | :--- |:-----------------------------------------------------------------------------------------------------------------|
| **Email** | Phishing Email Dataset | [Kaggle Link](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) | **Important:** Download the archive and extract ONLY the file named **`Enron.csv`**. Use this file for training. |
| **SMS** | SMS Spam Collection | [Kaggle Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Download the `SMSSpamCollection.csv` file (tab-separated).                                                       |
| **URL** | Phishing Site URLs | [Kaggle Link](https://www.kaggle.com/datasets/rakeshjampa/phiusiil-phishing-url-dataset) | Download `phishing_site_urls.csv`.                                                                               |

Run the main application script to access the interactive menu:

```bash
python main.py
```
Menu Options
1. Train Model:

   - Initiates the ETL pipeline.

   - Merges data, handles class imbalance, extracts TF-IDF features, and trains the Random Forest model.

   - Saves the trained model to models/phishing_model.pkl.

2. Test Model (Interactive Mode):

   - Option 1 (Email): Input a "Subject" and "Body" separately.

   - Option 2 (SMS/URL): Input raw text strings.

3. Quick Prediction:

   - Fast, single-line input for rapid classification checks.
---

**Language:** Python 3.8+
* **Data Manipulation:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn (Random Forest, TF-IDF, Pipeline)
* **NLP:** NLTK, Re (Regular Expressions)
* **Serialization:** Joblib
* **Visualization:** Matplotlib, Seaborn


This project was developed for the **Computer Science Project (CSEMCSPCSP01)** at **IU International University of Applied Sciences**.

* **Nmae:** Suprit Somashekaraiah
* **Matriculation Number:** 4250964
* **Date:** December 2025