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

## 🕒 Time of Tweet vs Sentiment Relationship

A heatmap analysis was conducted to examine the relationship between **Time of Tweet** and **Sentiment** categories.  
However, no significant correlation or observable trend was found.  

This indicates that **tweet timing (morning, afternoon, or night)** does not have a measurable impact on the expressed sentiment (positive, neutral, or negative).  
In other words, users’ emotional tone in tweets appears to be **independent of posting time**.

<img width="519" height="391" alt="download-1" src="https://github.com/user-attachments/assets/0d3ea4bb-7d1d-4cd7-8f59-fa29c0abc710" />

## 👤 Age of User vs Sentiment Relationship

A heatmap analysis was also performed to explore the relationship between **Age of User** and **Sentiment** categories.  
Similar to the time-based analysis, **no clear correlation or consistent pattern** was observed.  

This suggests that users’ **age groups** do not significantly influence the **emotional polarity** of their tweets.  
In summary, the distribution of positive, neutral, and negative sentiments appears **uniform across different age ranges**.

---

# 🧠 Sentiment Analysis – Model Comparison

This section presents the final evaluation results for three sentiment analysis approaches:  
**Linear SVM**, **Multinomial Naive Bayes**, and **VADER Lexicon-Based Analysis**.  
Each model is assessed using a consistent confusion matrix layout and performance metrics.

---

## ⚙️ 1. Linear SVM (TF-IDF + Age + Time + Country)

**Model Description:**  
This model combines textual TF-IDF features with categorical metadata such as user age, tweet time, and country.  
The classifier uses a **Linear SVM** with `class_weight="balanced"` to handle class imbalance.

**Confusion Matrix:**

<img width="579" height="490" alt="download-3" src="https://github.com/user-attachments/assets/06d9587a-3d67-4b23-be65-0ded57dd1b9f" />

**Classification Report:**

<img width="490" height="211" alt="SVM +Categorical" src="https://github.com/user-attachments/assets/0d5a02fa-ec3c-48fb-8e81-ab0429f078e6" />


**Interpretation:**  
- Diagonal dominance shows strong separation among positive, neutral, and negative classes.  
- Categorical context improved the model’s precision in detecting nuanced sentiments.  
- SVM’s margin-based optimization enhanced boundary clarity between classes.

---

## ⚙️ 2. Multinomial Naive Bayes (TF-IDF Only)

**Model Description:**  
A classic probabilistic model trained purely on **TF-IDF textual features**, capturing term frequency and document uniqueness.  
This serves as a baseline model for comparison.

**Confusion Matrix:**

<img width="578" height="490" alt="download-2" src="https://github.com/user-attachments/assets/81e017ca-7a17-47d5-9366-0e7a23ca5e0c" />


**Classification Report:**

<img width="474" height="219" alt="Ekran Resmi 2025-11-04 20 48 12" src="https://github.com/user-attachments/assets/730b3857-69bc-46f1-9872-9b49da32e661" />


**Interpretation:**  
- The model performs consistently across classes with ~67% overall accuracy.  
- Shows reliable F1-scores around **0.66–0.73**, performing best on **positive** sentiments.  
- Slight confusion between *neutral* and *positive* categories due to lexical similarity.

---

## ⚙️ 3. VADER (Lexicon-Based)

**Model Description:**  
A rule-based, no-training sentiment analyzer relying on lexical polarity scores from the **VADER sentiment lexicon**.  
Each text’s polarity is determined using a compound sentiment score threshold.

**Confusion Matrix:**

<img width="578" height="490" alt="download-4" src="https://github.com/user-attachments/assets/6aa05c0f-8fdc-40ae-91b4-c16bd87ea9d2" />


**Classification Report:**

<img width="479" height="222" alt="Ekran Resmi 2025-11-04 20 47 18" src="https://github.com/user-attachments/assets/9d904250-eafb-42e5-b062-b06642c53c60" />


**Interpretation:**  
- Achieved ~63% accuracy without any model training.  
- Strong recall for **positive** tweets (0.87) but lower precision due to overgeneralization.  
- Struggles with **neutral** tones, common for lexicon-based models where contextual cues are missed.  

---

## 📊 Overall Takeaways

| Model | Input Features | Accuracy | Strength |
|-------|----------------|-----------|-----------|
| **Linear SVM** | TF-IDF + Categorical | ~0.67 | Strong class separation, robust to imbalance |
| **Naive Bayes** | TF-IDF Only | ~0.67 | Fast, interpretable, solid baseline |
| **VADER** | Lexicon-Based | ~0.63 | No training, language-aware |

---

All models demonstrate competitive performance across sentiment classes.  
While **SVM** benefits from additional contextual metadata, **Naive Bayes** remains a lightweight yet reliable baseline.  
**VADER**, despite its simplicity, performs strongly without any training — highlighting the power of lexicon-based sentiment scoring for general-purpose text.
