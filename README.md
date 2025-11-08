# Fake News Detection using Ensemble Machine Learning

## Overview
Fake-News-Detection is a Streamlit-based web application that detects whether a news article is real or fake using ensemble machine learning techniques. It combines models such as Random Forest, Logistic Regression, SVM, Naive Bayes, Gradient Boosting, and KNN to provide high accuracy and reliability. The app uses TF-IDF for text vectorization and provides real-time predictions with detailed performance metrics.

---

## Features
- Detects fake or real news using ensemble ML models.
- Uses TF-IDF text vectorization.
- Combines multiple models for improved prediction accuracy.
- Displays accuracy metrics and confusion matrix visualization.
- Supports topic-wise datasets like business, politics, technology, and more.
- Interactive Streamlit web interface.

---

## Project Structure
Fake-News-Detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore file
├── datasets/                       # Folder for datasets
│   ├── business_data.csv
│   ├── technology_data.csv
│   ├── sports_data.csv
│   ├── entertainment_data.csv
│   ├── education_data.csv
│   ├── politics_data.csv
│   └── current_affairs_data.csv
└── models/                         # Trained models
    ├── ensemble_model.pkl
    └── ensemble_vectorizer.pkl
git clone https://github.com/Dhanush-1112-v/Fake-News-Detection.git

cd Fake-News-Detection


2. Create a virtual environment:


python -m venv venv
venv\Scripts\activate # For Windows

source venv/bin/activate # For macOS/Linux

3. Install dependencies:


pip install -r requirements.txt


4. Run the Streamlit app:


streamlit run app.py


5. Open your browser and visit the local URL (default: http://localhost:8501)

---

## Example Usage
Input:  
"Government launches new education policy for rural development."  
Output: ✅ Real News  

Input:  
"Aliens spotted controlling global news media from space."  
Output: 🚫 Fake News  

---

## Machine Learning Models Used
1. Random Forest Classifier  
2. Logistic Regression  
3. Support Vector Machine (SVM)  
4. Gradient Boosting Classifier  
5. Multinomial Naive Bayes  
6. K-Nearest Neighbors (KNN)  
7. Ensemble (Voting and Stacking Classifiers)

---

## Model Performance
| Model | Accuracy (%) | Description |
|--------|---------------|-------------|
| Random Forest | 85.2 | Balanced precision and recall |
| Logistic Regression | 83.7 | Fast and reliable baseline |
| SVM | 86.1 | High accuracy for text classification |
| Gradient Boosting | 87.5 | Excellent generalization |
| Naive Bayes | 81.3 | Effective for word-based data |
| KNN | 79.6 | Simple and interpretable |
| Ensemble (Voting) | 89.8 | Combined model, highest accuracy |

---

## Technologies Used
- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Joblib

---

## How It Works
1. Loads and preprocesses datasets using NLTK for stemming and stopword removal.
2. Text data is vectorized using TF-IDF.
3. Multiple models are trained individually.
4. Ensemble techniques (Voting/Stacking) combine model predictions.
5. The final ensemble model predicts whether a given news article is real or fake.

---

## Future Enhancements
- Integrate deep learning models (LSTM/BERT) for improved performance.
- Add real-time news scraping for verification.
- Deploy on Streamlit Cloud for public access.
- Provide explainability (why an article was classified as fake/real).

---

## License
This project is licensed under the MIT License.  
You are free to use, modify, and distribute this project for both personal and commercial purposes.  
See the LICENSE file for details.

---

## Author
Sai Dhanush Vanjari  
Email: dhanushvanjari@gmail.com  
GitHub: https://github.com/Dhanush-1112-v

---

## Acknowledgments
- Streamlit for providing a fast, easy-to-use web framework.
- NLTK and Scikit-learn for powerful NLP and ML tools.
- Open-source datasets and research projects that inspired this work.

---

⭐ If you found this project useful, please give it a star on GitHub!
---

## Installation and Setup
1. Clone this repository:
