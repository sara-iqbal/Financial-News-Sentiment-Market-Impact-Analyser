
# FinBERT Sentiment & Market Impact Analyzer

**NLP Pipeline for Financial News Sentiment and Stock Price Correlation**


Can AI accurately predict market sentiment from a single headline? This project bridges the gap between **Natural Language Processing (NLP)** and **Quantitative Finance**. By fine-tuning **FinBERT** on expert-labeled financial data, this system classifies news sentiment and correlates those findings with real-world next-day stock price movements.



---

## The Tech Stack

* **Base Model:** `ProsusAI/finbert` (HuggingFace)
* **Data Source:** Financial PhraseBank & Yahoo Finance (`yfinance`)
* **Deep Learning:** Transformers Trainer API (PyTorch)
* **Deployment:** Gradio & Hugging Face Spaces
* **Visualization:** Plotly (Interactive Charts)

---

## Performance at a Glance

The model achieves high precision across all sentiment classes, with particularly strong performance in identifying positive market signals.

| Metric | Score |
| --- | --- |
| **Test Accuracy** | **87.3%** |
| **F1 Macro** | **0.861** |
| **Positive Class F1** | **0.892** |

---

## Technical Architecture & Pipeline

### 1. Domain-Specific Fine-Tuning

Unlike standard BERT, **FinBERT** is pre-trained on 4.9B tokens of financial text (Reuters, SEC filings, etc.). This ensures the model understands specialized terminology like *EBITDA, margin compression,* and *covenant breaches*.

### 2. The "Compound Score" Advantage

Rather than using simple labels, I implemented a **Compound Sentiment Score**:


$$Compound = P(\text{positive}) - P(\text{negative})$$


This preserves the "intensity" of a headline, allowing for a more granular correlation with continuous price movements.

### 3. Market Correlation Analysis

After inference, compound scores are compared against next-day stock returns bucketed into quintiles. The result? **A statistically meaningful positive correlation** between FinBERT scores and next-day price direction.

---

##  Deployment & Usage

The live inference engine is deployed on **Hugging Face Spaces**. Anyone can input a financial headline to receive:

1. **Sentiment Probability:** (Positive/Neutral/Negative)
2. **Market Sentiment Score:** (The gradient of the news impact)
3. **Estimated Impact:** Directional signal based on historical correlation.

### How to Run Locally

```bash
git clone https://github.com/YourUsername/finbert-sentiment-analyser.git
cd finbert-sentiment-analyser
pip install -r requirements.txt
python app.py

```

---

##  Repository Structure

* `finbert_sentiment_colab.ipynb`: The complete training and correlation pipeline.
* `app.py`: Gradio interface script for production deployment.
* `requirements.txt`: Environment dependencies.

---

## About the Author

**Sara**

* **MSc Data Science** | **B.Tech AI & Machine Learning**
* 📍 London, UK

---

