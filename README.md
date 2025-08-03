# 📱 Spam SMS Detection Web App using Streamlit & Machine Learning

An interactive *SMS Spam Classifier Web App* built with *Python and Streamlit*.  
This project allows users to enter an SMS message and instantly get a prediction on whether it's *Spam* or *Not Spam (Ham)* using a trained Machine Learning model.

---

## 🚀 Overview

Spam messages are a major issue, leading to scams, frauds, and unnecessary distractions.  
This project demonstrates how *Natural Language Processing (NLP)* and *Machine Learning* techniques can be used to detect spam SMS messages effectively.

The project has two main components:
1. *Model Training Notebook (.ipynb)* — Preprocesses data, trains a classifier, and saves the model.
2. *Streamlit Web App (.py)* — Provides a user-friendly interface to test any SMS message in real-time.

---

## ✨ Features
- 🔍 Real-time *Spam SMS Detection*.
- 🧹 Automatic *Text Cleaning & Preprocessing*.
- ⚡ Lightweight and Fast Web App using *Streamlit*.
- 💾 Model and Vectorizer loading from pre-trained .pkl files.
- 🎯 High accuracy using *Multinomial Naive Bayes* and *TF-IDF Vectorization*.

---

## 🛠 Installation & Setup Instructions

### 🚀 Run Locally (Streamlit App)
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/spam-sms-detector.git
   cd spam-sms-detector
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Ensure the following model files are present in the project folder:

tfidf_vectorizer.pkl

spam_classifier_model.pkl

label_encoder.pkl

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Open the local URL in your browser to interact with the app.

🧑‍💻 Usage Instructions
Type or paste an SMS message in the input box.

Click Enter or Submit.

The app will predict and display whether the message is SPAM or Not Spam (Ham).

🧰 Technologies Used
Python 3.x

Pandas & NumPy

Scikit-learn

NLTK (Stopwords)

Imbalanced-learn (SMOTE)

Streamlit (Web Interface)

Joblib (Model Serialization)

Regex (Text Cleaning)

📂 Folder Structure
bash
Copy
Edit
spam-sms-detector/
│
├── app.py                      # Streamlit App Interface (Web UI)
├── model_training.ipynb        # Jupyter Notebook for Model Training
├── tfidf_vectorizer.pkl        # Saved TF-IDF Vectorizer
├── spam_classifier_model.pkl   # Trained Spam Detection Model
├── label_encoder.pkl           # Label Encoder for Labels (Spam/Ham)
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation

Inspired by common NLP Spam Detection workflows and extended for web deployment using Streamlit.

📄 License
This project is licensed under the MIT License — free to use, modify, and distribute.

⭐ Feedback & Contributions
If you found this project helpful, give it a ⭐ on GitHub!

Feel free to open Issues or submit Pull Requests for improvements or feature requests.



---

## 🙋‍♂ Author:

   Name: Afzal Shah
   GitHub Profile: https://github.com/syed-afzal-shah
   Linkedin Profile: https://www.linkedin.com/in/afzal-shah-ai-engineer

---
