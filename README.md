# 🧠 Sentiment Analysis on YouTube Comments

This project performs **sentiment analysis** on YouTube video comments using both traditional machine learning techniques (like Random Forest with TF-IDF) and modern transformer-based models (HuggingFace Transformers). It processes, cleans, and classifies comments into positive, negative, or neutral sentiments, enabling meaningful insights from user-generated content.

---

## 📂 Dataset Source

- **YouTube Comments** are fetched via the **YouTube Data API v3** using video IDs.
- Can be extended to include datasets from other platforms like Twitter, Reddit, etc.

---

## 🔑 Key Features

- 🔍 **YouTube Data Extraction** using Google API.
- 🧼 **Text Cleaning & Preprocessing** with regular expressions and NLTK.
- 📊 **TF-IDF + Random Forest** pipeline for classical sentiment classification.
- 🤖 **Transformers (e.g., BERT-based models)** for advanced NLP analysis.
- 🌫️ **WordCloud Visualization** for frequently used terms.
- 📈 Evaluation using **Confusion Matrix** and **Classification Report**.
- 🔧 Easily customizable pipeline for future datasets.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Pandas & NumPy** | Data manipulation and analysis |
| **Matplotlib & Seaborn** | Data visualization |
| **NLTK** | Natural Language Toolkit for preprocessing |
| **Scikit-learn** | ML model training and evaluation |
| **HuggingFace Transformers** | Pretrained transformer models for sentiment classification |
| **Google API Client** | Fetching comments from YouTube |
| **WordCloud** | Visualizing text data |

---

## 🔄 Process Workflow

1. **Data Collection**
   - Use YouTube Data API to collect video comments.
   - Save data into DataFrame format.

2. **Preprocessing**
   - Remove URLs, emojis, special characters.
   - Tokenization, stopword removal (NLTK).

3. **Modeling**
   - TF-IDF Vectorization → Random Forest Classifier.
   - Transformers pipeline with `AutoTokenizer` and `AutoModelForSequenceClassification`.

4. **Evaluation**
   - Use classification report & confusion matrix for evaluation.
   - Visualize word frequencies with WordCloud.

5. **Visualization**
   - Graphical analysis of sentiment distribution.
   - Word clouds for positive/negative sentiment clusters.

---

## 📌 Use Cases

- Brand reputation monitoring
- Product review analysis
- Public opinion tracking
- YouTube content feedback analysis

---

## 📍 Future Improvements

- Add support for multilingual comments.
- Integrate with Twitter and Reddit APIs.
- Fine-tune transformer models for specific domains.
- Real-time sentiment dashboard using Streamlit or Flask.

---

## 🙌 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
