# Social Media Misinformation Detector (SMD)

## Overview

The Social Media Misinformation Detector (SMD) is an AI-powered system designed to identify potentially misleading social media posts and provide evidence-based explanations for its predictions.
With the rapid spread of information on platforms such as X (Twitter), Facebook, Instagram, and YouTube, misinformation can influence public opinion, create panic, spread health-related myths, 
and promote scams. This project aims to automatically detect such content using Machine Learning and Natural Language Processing (NLP).

---

## Problem Statement

Build an intelligent system capable of:

* Analyzing social media posts
* Detecting misinformation patterns
* Estimating credibility scores
* Classifying content into:

  * Reliable
  * Suspicious
  * Misinformation
* Generating explainable evidence reports

---

## Features

### Stage 1: Text Analysis

* Text preprocessing
* Feature extraction
* Sentiment analysis
* Detection of sensational language
* Detection of excessive punctuation
* Detection of all-caps emphasis

### Stage 2: Misinformation Classification

* TF-IDF Vectorization
* Logistic Regression Classifier
* Multi-class prediction:

  * Reliable
  * Suspicious
  * Misinformation

### Stage 3: Evidence Report Generation

Generates a structured report containing:

* Classification
* Credibility score
* Confidence score
* Class probabilities
* Evidence signals

### Stage 4: Interactive Web Application

Built using Streamlit:

* User-friendly interface
* Real-time prediction
* Downloadable JSON reports
* Visual probability charts

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* NLP Techniques
* Joblib

---

## Project Structure

```text
Social-Media-Misinformation-Detector/
│
├── data_analysis.ipynb
├── feature_extraction.ipynb
├── model_training.ipynb
├── Evidence-Report-generation.ipynb
├── 05_streamlit_app.py
│
├── models/
│   ├── smd_logistic_regression_model.pkl
│   ├── smd_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│
├── train.tsv
├── valid.tsv
├── test.tsv
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/komal200515/socialmedia-misinfo-detection.git
cd socialmedia-misinfo-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python -m streamlit run 05_streamlit_app.py
```

---

## Example Input

```text
Breaking News!!! Doctors found a MIRACLE cure. Share urgently!!!
```

### Example Output

```json
{
  "classification": "Suspicious",
  "credibility_score": 0.464,
  "confidence": "46.4%",
  "evidence": [
    {
      "signal": "sensational_language"
    },
    {
      "signal": "excessive_punctuation"
    }
  ]
}
```

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Future Improvements

* BERT / RoBERTa based classification
* Fact-checking using trusted sources
* Explainable AI dashboard
* Advanced NLP pipelines
* Real-time social media integration

---

## Team Member

* Komal Mahawar


---
