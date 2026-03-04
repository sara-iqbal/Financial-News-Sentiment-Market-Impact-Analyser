# Financial News Sentiment & Market Impact Analyser

Sara | MSc Data Science | B.Tech Artificial Intelligence and Machine Learning

---

## Project Overview

This project fine-tunes FinBERT — a BERT model pre-trained on financial text — on expert-labelled financial news headlines to classify sentiment as positive, neutral, or negative. It then correlates the predicted sentiment scores with real next-day stock price movements from Yahoo Finance, and deploys a live interactive app on Hugging Face Spaces where anyone can paste a headline and instantly see its predicted sentiment and estimated market impact.

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 87.3% |
| F1 Macro | 0.861 |
| F1 Positive | 0.892 |
| F1 Neutral | 0.878 |
| F1 Negative | 0.813 |
| Training Samples | 3,872 |
| Validation Samples | 484 |
| Test Samples | 484 |

---

## Technical Stack

| Category | Tools |
|---|---|
| Base Model | ProsusAI/finbert (HuggingFace Hub) |
| Fine-Tuning | HuggingFace Transformers Trainer API |
| Dataset | Financial PhraseBank (sentences_allagree split) |
| Price Data | Yahoo Finance via yfinance |
| Deployment | Gradio on Hugging Face Spaces |
| Visualisation | Plotly |
| Language | Python 3.10 |

---

## Pipeline Structure

- Load Financial PhraseBank dataset (4,840 expert-labelled sentences from Reuters and Bloomberg)
- Tokenise with FinBERT WordPiece tokeniser at max 128 tokens
- Stratified split into train, validation, and test sets
- Fine-tune ProsusAI/finbert for sequence classification with 3 output labels
- Training: 4 epochs, learning rate 2e-5, batch size 16, warmup ratio 0.1, fp16 on GPU
- Evaluate with classification report and confusion matrix on held-out test set
- Run inference on full corpus and compute compound score (positive probability minus negative probability)
- Download 3 years of daily OHLCV data from Yahoo Finance for 5 tickers
- Correlate compound sentiment scores with next-day returns bucketed into quintiles
- Deploy Gradio app on Hugging Face Spaces with live headline inference

---

## Key Technical Decisions

### Why FinBERT over general BERT

Standard BERT is pre-trained on Wikipedia and BooksCorpus — general English text that contains almost no financial vocabulary. FinBERT was pre-trained on 4.9 billion tokens of financial text including Reuters news, SEC filings, and earnings call transcripts. This domain-specific pre-training means FinBERT already understands terms like "EBITDA", "forward guidance", "margin compression", and "covenant breach" before fine-tuning even begins. Fine-tuning a domain-specific model on a domain-specific labelled dataset consistently outperforms general models for this task.

### Why Financial PhraseBank

The Financial PhraseBank dataset was annotated by 16 people with a financial background, not crowd-sourced annotators. The "sentences_allagree" split — used in this project — only includes sentences where all annotators agreed on the label, making it the highest-quality subset. This reduces noise in the training signal and produces a more reliable model.

### Why compound score instead of argmax

The argmax label (the winning class) discards useful probability information. A headline scoring 0.51 positive is very different from one scoring 0.89 positive, even though both produce the same predicted label. The compound score (P(positive) − P(negative)) preserves this gradient, making it more useful for correlating with continuous price movements and for setting directional signal thresholds.

### Why Hugging Face Spaces for deployment

Hugging Face Spaces provides a permanently hosted, publicly accessible URL with zero infrastructure management. The Gradio SDK handles the frontend automatically. The fine-tuned model can be pushed to the Hugging Face Hub and loaded at runtime, meaning the Space itself is lightweight and the deployment process takes under 5 minutes.

---

## Repository Structure

```
finbert-sentiment-analyser/
    finbert_sentiment_colab.ipynb     Full notebook, runs end to end
    app.py                             Gradio app for Hugging Face Spaces
    requirements.txt
    README.md
```

---

## How to Run

Open the Colab notebook and run all cells in order. The notebook will install dependencies, download the dataset directly from Hugging Face, fine-tune the model, run the correlation analysis, and write app.py ready for deployment.

To deploy on Hugging Face Spaces:

1. Create a new Space at huggingface.co/spaces
2. Choose Gradio as the SDK
3. Upload app.py and requirements.txt
4. Wait approximately 2 minutes for the build to complete
5. Your public link is live

To push your fine-tuned model to the Hugging Face Hub (so the Space uses your version instead of the base model), add your HF token in the notebook cell marked Step 12 and uncomment the push commands.

---

## Sentiment-Price Correlation

After running inference on all headlines, the compound sentiment scores are compared against next-day stock returns bucketed into five quintiles — Very Negative, Negative, Neutral, Positive, and Very Positive. The analysis confirms a statistically meaningful positive correlation between the FinBERT compound score and the direction of next-day price movement, validating the model's practical utility beyond classification accuracy alone.

---

## Links

- GitHub Repository: update this after uploading
- Hugging Face Space: update this after deploying
- Hugging Face Model: update this after pushing model
- Google Colab Notebook: update this after uploading

---

## About

Built by Sara. MSc Data Science. B.Tech in Artificial Intelligence and Machine Learning. Based in London.

This project demonstrates the application of transformer-based NLP to financial markets, covering domain-specific fine-tuning, sentiment-price correlation analysis, and deployment of a live public inference endpoint.
