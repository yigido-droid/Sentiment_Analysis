# Comparison between ML-Based and Lexicon-Based Approaches in Sentiment Analysis

## 📌 Overview
This project performs **Sentiment Analysis** on Twitter-like text data using both **Machine Learning (ML)** and **Lexicon-Based** approaches.  
The goal is to classify tweets into **positive**, **negative**, and **neutral** sentiments using textual features extracted from tweet content.

---

## Why TF-IDF and Word2Vec Representations?

Before model training, text data must be transformed into numerical vectors that machine learning algorithms can process.  
Two key representations are used in this project: **TF-IDF** and **Word2Vec**.

### 🧩 TF-IDF (Term Frequency – Inverse Document Frequency)
- Measures **how frequent** and **how unique** a word is in the dataset.  
- Focuses on **term importance**, but ignores **semantic meaning** or relationships between words.  
- Ideal for models like **Naive Bayes**, which rely on frequency-based probabilities.

### 🌐 Word2Vec (Semantic Embeddings)
- Learns **contextual meaning** of words by analyzing their co-occurrence in text.  
- Produces **dense vector embeddings** where similar words (e.g., “happy”, “joyful”) are close in space.  
- Each document is represented by the **average of its word embeddings**, capturing overall semantic tone.

> 🧠 In short, **TF-IDF** captures *word importance*, while **Word2Vec** captures *word meaning*.

---

## 📂 Dataset
**Source:** `Datasets/train_dataset.csv`  
**Cleaned File:** `clean_train.csv`

After removing irrelevant columns (`selected_text`, `Population -2020`, `Land Area (Km²)`, `Density (P/Km²)`),  
the dataset contains **27,481 records** and **6 columns**.

### 🧾 Sentiment Distribution
| Sentiment | Count | Percentage |
|------------|--------|-------------|
| Neutral | 11,118 | 40.46% |
| Positive | 8,582 | 31.23% |
| Negative | 7,781 | 28.31% |

---

## 🔍 Exploratory Data Analysis (EDA)

### 🕒 Time of Tweet vs Sentiment
A heatmap analysis revealed **no significant correlation** between tweet timing and sentiment.  
Users’ emotional tones are **independent of posting time**.

<img width="541" height="390" alt="Ekran Resmi 2025-11-10 18 01 19" src="https://github.com/user-attachments/assets/b8bff2d7-9b64-4fc4-9aa3-b4a2ae007e54" />


### 👤 Age of User vs Sentiment
Similarly, **no pattern** was found between user age and sentiment polarity.  
Positive, neutral, and negative sentiments are evenly distributed across age groups.

<img width="536" height="395" alt="Ekran Resmi 2025-11-10 18 01 27" src="https://github.com/user-attachments/assets/9e8119e5-ee4e-4438-9970-ab2ea5a4ec70" />


Hence, **text content** itself remains the primary determinant of sentiment.

---

# 🧠 Sentiment Analysis – Model Results

Below are the final evaluation results for three sentiment analysis approaches.

---

## ⚙️ 1. Word2Vec Embeddings + Linear SVM

**Model Description:**  
This model represents tweets using **Word2Vec embeddings** trained on the dataset itself.  
Each tweet vector is the average of its word embeddings, which capture **contextual and semantic meaning**.  
A **Linear SVM** classifier is trained on these dense vectors with `class_weight="balanced"`.

**Confusion Matrix:**

<img width="581" height="487" alt="Word2Vec Confusion Matrix" src="https://github.com/user-attachments/assets/c3523792-f1d8-40e6-9740-622a7587aa6a" />

**Classification Report:**

<img width="469" height="221" alt="Word2Vec Classification Report" src="https://github.com/user-attachments/assets/637c0c9d-9256-4a71-bb87-07f03499df85" />

**Interpretation:**  
- Achieved **~61.7% accuracy** across three classes.  
- Captures **semantic similarity** (e.g., *great → good → amazing*) beyond surface-level word counts.  
- Slightly lower accuracy than TF-IDF models, but offers better **contextual understanding**.  
- Future work can include **pre-trained embeddings** (e.g., GoogleNews, GloVe) to enhance results.

---

## ⚙️ 2. Multinomial Naive Bayes (TF-IDF Only)

**Model Description:**  
A classic probabilistic model trained on **TF-IDF features** representing each tweet’s word frequency and uniqueness.  
It serves as a **baseline model** for comparison.

**Confusion Matrix:**

<img width="578" height="490" alt="Naive Bayes Confusion Matrix" src="https://github.com/user-attachments/assets/81e017ca-7a17-47d5-9366-0e7a23ca5e0c" />

**Classification Report:**

<img width="474" height="219" alt="Naive Bayes Classification Report" src="https://github.com/user-attachments/assets/730b3857-69bc-46f1-9872-9b49da32e661" />

**Interpretation:**  
- Achieved **~67% accuracy**, performing best on **positive** sentiments.  
- Confusion between *neutral* and *positive* due to overlapping vocabulary.  
- Fast, interpretable, and effective — a strong baseline for text classification tasks.

---

## ⚙️ 3. VADER (Lexicon-Based)

**Model Description:**  
A **rule-based sentiment analyzer** that uses the **VADER lexicon** to classify tweets without any training.  
It assigns polarity based on pre-defined word sentiment scores.

**Confusion Matrix:**

<img width="578" height="490" alt="VADER Confusion Matrix" src="https://github.com/user-attachments/assets/6aa05c0f-8fdc-40ae-91b4-c16bd87ea9d2" />

**Classification Report:**

<img width="479" height="222" alt="VADER Classification Report" src="https://github.com/user-attachments/assets/9d904250-eafb-42e5-b062-b06642c53c60" />

**Interpretation:**  
- Achieved **~63% accuracy** without any machine learning.  
- Strong recall for **positive** tweets (0.87) but weaker precision for **neutral** tones.  
- Limited in handling **context**, **sarcasm**, or **negation** compared to ML models.

---

## 📊 Overall Comparison

| Model | Vectorization | Accuracy | Key Strength |
|-------|----------------|-----------|---------------|
| **Word2Vec + SVM** | Dense semantic embeddings | **~0.62** | Captures contextual meaning and word similarity |
| **Naive Bayes (TF-IDF)** | Sparse TF-IDF vectors | **~0.67** | Fast, interpretable, frequency-based baseline |
| **VADER (Lexicon)** | Predefined polarity lexicon | **~0.63** | No training, quick and language-aware |

---

## 🧩 Conclusion

- **TF-IDF + Naive Bayes** achieved the highest accuracy, showing that statistical term weighting remains effective for short-text sentiment classification.  
- **Word2Vec + SVM** provided a deeper understanding of context and meaning, useful for semantic-rich datasets.  
- **VADER**, though simple, proved competitive with zero training cost.  

In conclusion, **hybrid use of statistical and semantic features** offers a balanced trade-off between accuracy, interpretability, and linguistic depth — essential for robust sentiment analysis systems.
