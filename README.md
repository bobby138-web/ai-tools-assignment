(Scikit-learn + TensorFlow + spaCy + Streamlit):

# 🤖 AI Tools Assignment

This repository contains the complete implementation and documentation for my **AI Tools Assignment**, showcasing applied machine learning, deep learning, and natural language processing using popular Python libraries like **Scikit-learn**, **TensorFlow**, and **spaCy** — along with model deployment via **Streamlit**.

---

## 📘 Project Overview

The assignment is divided into three major parts:

### **1. Theoretical Understanding (40%)**
- Explained the key differences between **TensorFlow** and **PyTorch**.
- Described **use cases of Jupyter Notebooks** in AI development.
- Illustrated how **spaCy** improves NLP tasks.
- Compared **Scikit-learn** and **TensorFlow** in terms of target use, ease of use, and community support.

🧾 A detailed PDF report with all theoretical answers is included:


📄 AI_Tools_Assignment_Complete.pdf


---

### **2. Practical Implementation (50%)**

#### 🧩 Task 1 — Classical ML with Scikit-learn
**Dataset:** Iris Species  
**Goal:** Predict iris flower species using a Decision Tree Classifier.

**Key Steps:**
- Preprocessed the dataset (handled missing values, label encoding)
- Trained and evaluated a Decision Tree Classifier
- Measured accuracy, precision, and recall

#### 🧠 Task 2 — Deep Learning with TensorFlow
**Dataset:** MNIST Handwritten Digits  
**Goal:** Train a Convolutional Neural Network (CNN) to classify digits.

**Model Features:**
- Built using `tensorflow.keras`
- Achieved >95% accuracy on test data
- Visualized predictions for sample images
- Deployed the model using **Streamlit**

#### 💬 Task 3 — NLP with spaCy
**Dataset:** Amazon Product Reviews  
**Goal:** Perform **Named Entity Recognition (NER)** and **Sentiment Analysis**

**Approach:**
- Extracted entities (product names, brands)
- Used a rule-based sentiment analysis method to label reviews as positive or negative

---

### **3. Ethics & Optimization (10%)**

- **Bias Analysis:** Identified potential biases in MNIST and Amazon Reviews datasets.  
- **Fairness Tools:** Suggested using TensorFlow Fairness Indicators and spaCy’s rule-based improvements to reduce model bias.  
- **Debugging Challenge:** Fixed TensorFlow issues (e.g., dimension mismatches, wrong loss functions).  

---

## 🚀 Deployment (Bonus Task)
A simple **Streamlit** web app was developed to deploy the trained MNIST classifier.

Run locally using:
```bash
streamlit run app.py


If you’re using VS Code:

Activate your virtual environment

venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Launch the app and open your browser to http://localhost:8501

🧰 Tech Stack
Category	Tools Used
Classical ML	Scikit-learn
Deep Learning	TensorFlow / Keras
NLP	spaCy
Deployment	Streamlit
Environment	Python 3.12, VS Code
Visualization	Matplotlib, Seaborn
📂 Repository Structure
ai-tools-assignment/
│
├── mnist_cnn.py                # TensorFlow CNN model for MNIST
├── iris_decision_tree.py       # Scikit-learn Decision Tree model
├── nlp_spacy.py                # spaCy NER + sentiment analysis
├── app.py                      # Streamlit deployment script
│
├── AI_Tools_Assignment_Complete.pdf  # Final report (theoretical + ethics)
├── requirements.txt
├── README.md                   # This file
└── data/                       # (optional) Local datasets

✍️ Author

Prepared by: Fredric Bobby and giremunga

🏁 Acknowledgements

Special thanks to:

TensorFlow and PyTorch developer communities

The Scikit-learn and spaCy teams for their open-source contributions

Streamlit for simplifying AI deployment

📜 License

This project is for educational and academic purposes only.
© 2025 Fredric Bobby. All rights reserved.
