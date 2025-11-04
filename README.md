# Comparison between ML-Based approach and Lexicon-Based approach in Sentiment Analysis

## 📌 Overview
This project performs **Sentiment Analysis** on Twitter-like text data using both **Machine Learning** and **Lexicon-Based** approaches.  
The goal is to classify tweets into **positive**, **negative**, and **neutral** sentiments using textual and categorical features such as:

- Tweet text  
- Time of Tweet  
- Age of User  
- Country

## Why TF-IDF Transformation?

Before training the models, all text is transformed using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF is used because:

- It converts raw text into **numeric vectors** that machine learning models can understand.
- It measures how **important** a word is in a document relative to the whole dataset.
- Common words (e.g., *"the", "and", "to"*) receive **lower weight**, while more **unique / meaningful words** that help distinguish sentiment (e.g., *"amazing", "terrible"*) receive **higher weight**.
- Compared to Bag-of-Words, TF-IDF prevents very frequent but uninformative words from dominating model decisions.

Mathematically:

> **TF (Term Frequency):** How often a word appears in a document  
> **IDF (Inverse Document Frequency):** How rare that word is across all documents

By emphasizing **unique usage frequency**, TF-IDF enables the classifier to focus on sentiment-carrying expressions.

---

## Model Comparison

The project compares the performance of:

- **Linear SVM (TF-IDF + categorical features)**  
  ➤ To study the impact of categorical variables (*Age of User*, *Time of Tweet*, *Country*) when combined with text features under a **powerful classifier** that can capture complex decision boundaries.

- **Multinomial Naive Bayes (TF-IDF only)**  
  ➤ To understand how a **probabilistic model** behaves when using only text frequency information, letting us observe the effect of likelihood-based classification on sentiment prediction.

- **VADER (Lexicon-based sentiment analyzer)**  
  ➤ To compare automated ML models with a **rule-based lexicon method** that requires no training and classifies sentiment based on word-level polarity scores.

The project compares the performance of:
- **Linear SVM (TF-IDF + categorical features)**  
- **Multinomial Naive Bayes (TF-IDF only)**  
- **VADER (Lexicon-based sentiment analyzer)**  

---

## 📂 Dataset
**Source:** `Datasets/train_dataset.csv`  
**Final cleaned file:** `clean_train.csv`

After removing unnecessary columns:
- `selected_text`  
- `Population -2020`  
- `Land Area (Km²)`  
- `Density (P/Km²)`  

the dataset contains **27,481 rows** and **6 columns**.

### 🧾 Sentiment Distribution
| Sentiment | Count | Percentage |
|------------|--------|-------------|
| Neutral | 11,118 | 40.46% |
| Positive | 8,582 | 31.23% |
| Negative | 7,781 | 28.31% |

---

## 🔍 Exploratory Data Analysis (EDA)
Visual relationships were explored through heatmaps:

<img width="519" height="391" alt="download" src="https://github.com/user-attachments/assets/25c53ec6-16ee-47a8-b27c-5f1e3963a23f" />

<img width="519" height="391" alt="download-1" src="https://github.com/user-attachments/assets/0d3ea4bb-7d1d-4cd7-8f59-fa29c0abc710" />

---

## Models
Model 1(TF-IDF + SVM + Categorical):






