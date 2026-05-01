# Comparison between ML-Based, Lexicon-Based and Hybrid-Based Approaches in Sentiment Analysis

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

<img width="614" height="486" alt="Ekran Resmi 2025-11-10 18 05 57" src="https://github.com/user-attachments/assets/e47f92ea-b3ab-4465-95c6-4c7f1a97bb43" />

**Classification Report:**

<img width="488" height="225" alt="Ekran Resmi 2025-11-10 18 05 46" src="https://github.com/user-attachments/assets/b1021a7e-fc3d-407d-b669-27a58ab44427" />
  
**Interpretation:**  
- The model achieved an overall **accuracy of 61.4%**, performing moderately well across all sentiment categories.  
- **Positive tweets** had the highest precision (0.681), showing that the model effectively recognized optimistic language patterns.  
- **Neutral tweets** were the most challenging, often confused with nearby emotional tones (positive or negative).  
- **Negative tweets** were predicted with reasonable balance but had slightly lower recall, indicating some false negatives.  

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

---

## ⚙️ 4. Hybrid Model: BERT Embeddings + VADER Lexicon Features + Linear SVM

**Model Description:**  
The hybrid model combines **BERT-based contextual embeddings** with **VADER lexicon-based sentiment features**. Unlike the previous models, this approach does not rely only on statistical word frequencies, semantic word embeddings, or predefined sentiment rules. Instead, it combines two complementary sources of information:

- **BERT** captures the contextual and semantic meaning of each tweet.
- **VADER** provides explicit sentiment polarity scores and tweet-level linguistic features.

In this model, BERT is used as an **embedding extractor**, not as a direct classifier. For each tweet, the `[CLS]` token representation is extracted from BERT, producing a **768-dimensional contextual embedding**. This vector represents the overall semantic meaning of the tweet.

At the same time, VADER extracts lexicon-based and surface-level features, including:

- Negative sentiment score
- Neutral sentiment score
- Positive sentiment score
- Compound sentiment score
- Tweet length
- Word count
- Exclamation mark count
- Question mark count
- Uppercase ratio
- Elongated word count
- Hashtag count
- Mention count

These two feature groups are concatenated into a single hybrid feature vector:

**Confusion Matrix:**

<img width="594" height="488" alt="Ekran Resmi 2026-05-01 13 49 36" src="https://github.com/user-attachments/assets/85ea425f-e07c-4e94-bd06-bf478bead43f" />

**Classification Report:**

<img width="510" height="222" alt="Ekran Resmi 2026-05-01 13 49 18" src="https://github.com/user-attachments/assets/768f9983-6f7d-46a4-8be8-5770bf3101a6" />

**Interpretation:**
The model performed best on the **positive** class, with a precision of **0.734**, recall of **0.761**, and F1-score of **0.747**. This shows that positive tweets were captured effectively by combining BERT’s contextual embeddings with VADER’s polarity features.

The **negative** class also performed well, reaching an F1-score of **0.700**. The model correctly classified **1,120 negative tweets**, although some weaker negative expressions were confused with neutral tweets.

The **neutral** class was the most difficult to classify, with an F1-score of **0.660**. This is expected because neutral tweets often overlap with weakly positive or weakly negative language, making them harder to separate clearly.

---

## 📊 Overall Evaluation

The project compares four different sentiment analysis approaches: **Word2Vec + Linear SVM**, **TF-IDF + Multinomial Naive Bayes**, **VADER**, and the proposed **BERT + VADER + Linear SVM hybrid model**.

| Model | Approach Type | Representation / Features | Accuracy | Key Strength |
|---|---|---|---:|---|
| **Word2Vec + Linear SVM** | ML-Based | Dense semantic word embeddings | **61.4%** | Captures semantic similarity between words |
| **Multinomial Naive Bayes (TF-IDF)** | ML-Based | Sparse TF-IDF vectors | **~67%** | Fast, interpretable, and effective baseline |
| **VADER** | Lexicon-Based | Predefined sentiment polarity lexicon | **~63%** | Requires no training and provides direct polarity scores |
| **BERT + VADER + Linear SVM** | Hybrid-Based | Contextual BERT embeddings + VADER lexicon features | **69.94%** | Combines semantic context with explicit sentiment polarity signals |

Among all tested models, the **hybrid BERT + VADER + Linear SVM model achieved the highest overall accuracy**, reaching **69.94%**. This indicates that combining contextual language representation with lexicon-based sentiment features improves classification performance.

The **TF-IDF + Multinomial Naive Bayes** model performed strongly as a traditional machine learning baseline, achieving approximately **67% accuracy**. This shows that frequency-based word importance remains effective for short-text sentiment classification.

The **VADER lexicon-based approach** achieved approximately **63% accuracy** without requiring any model training. This demonstrates that predefined sentiment polarity rules can be useful for social media sentiment analysis, especially for clearly positive or negative tweets. However, VADER is limited because it cannot fully understand context, sarcasm, or subtle expressions.

The **Word2Vec + Linear SVM** model achieved **61.4% accuracy**. Although Word2Vec captures semantic similarity between words, representing each tweet by averaging word embeddings may cause loss of sentence-level meaning and word order information.

Overall, the hybrid model provided the strongest result because it combines two complementary sources of information. **BERT** captures contextual and semantic meaning, while **VADER** contributes explicit sentiment polarity indicators. As a result, the final Linear SVM classifier benefits from both deep language representation and lexicon-based sentiment knowledge.

---

## 🧩 Conclusion

This project investigated **machine learning-based**, **lexicon-based**, and **hybrid-based** approaches for sentiment analysis on Twitter-like text data. The classification task was defined as a three-class problem: **negative**, **neutral**, and **positive** sentiment prediction.

The results show that traditional machine learning models can still perform effectively on short-text sentiment classification. In particular, **TF-IDF + Multinomial Naive Bayes** produced a strong baseline result, showing that word frequency and term importance are useful indicators for sentiment prediction.

The **Word2Vec + Linear SVM** model introduced semantic word representation by capturing relationships between words. However, because tweet vectors were created by averaging word embeddings, the model may have lost important sentence-level structure and contextual details.

The **VADER lexicon-based model** showed that sentiment classification can be performed without supervised training by using predefined polarity scores. Although VADER performed competitively, it was less effective in handling contextual meaning, ambiguous sentiment, and neutral expressions.

The proposed **hybrid BERT + VADER + Linear SVM model** achieved the best performance, with an accuracy of **69.94%**. This result suggests that combining **contextual embeddings** with **lexicon-based sentiment features** creates a richer and more effective representation for sentiment classification.

In conclusion, the hybrid approach offers the most balanced solution among the tested methods. It benefits from BERT’s ability to understand contextual meaning and VADER’s ability to provide direct sentiment polarity signals. Therefore, combining deep semantic representations with lexicon-based features can improve sentiment analysis performance, especially for noisy and short social media texts.


