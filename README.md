# 🎬 IMDb Review Sentiment & Rating Predictor

A powerful Streamlit-based web app that predicts:
- 🎭 Whether a movie review is **Positive** or **Negative**
- 🌟 Gives an approximate **rating** (on a scale of 1 to 10)

Built using **Logistic Regression** for sentiment classification and **Linear Regression** for rating prediction — trained on the official [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).



## 📊 Model Performance

| Task                | Metric                | Score     |
|---------------------|------------------------|-----------|
| Sentiment Analysis  | Accuracy               | **~88%**  |
| Rating Prediction   | Mean Absolute Error    | **~0.61** |


## 🖥️ Demo UI

### ✅ Positive Review Output
<img src="UI_scrnshot\pos_review.png" width="700"/>

### ❌ Negative Review Output
<img src="UI_scrnshot\neg_review.png" width="700"/>

---

## 🚀 Features

- Sleek UI, IMDb branding and intuitive layout
- Dynamic star rating and rating slider
- Stores a mini history of predictions
- Styled sentiment labels: **green for positive**, **red for negative**

---

## 🛠️ How to Run

1. Install dependencies:
```bash
pip install -r req.txt
```

2. Make sure model_sentiment.pkl and model_rating.pkl are trained and saved using the training script.

3. Run the app
```bash
streamlit run app1.py
```

## 📂 Project Structure

```bash
├── app1.py                   # Streamlit UI
├── train_model.py            # Model training script
├── model_sentiment.pkl       # Saved sentiment classification model
├── model_rating.pkl          # Saved rating regression model
├── imdb_logo.png             # IMDb logo used in the UI
├── bg.jpg                    # Background image
├── screenshots/
│   ├── positive_output.png   # Screenshot of positive review
│   └── negative_output.png   # Screenshot of negative review
└── README.md

```
## 🧠 Models

    Sentiment: LogisticRegression

    Rating: LinearRegression

Both trained on labeled IMDb movie reviews (where rating is derived from the filename, as in 123_8.txt).

## 🙌 Acknowledgments

1. [IMDb Dataset, Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/)   

2. Streamlit Team

3. Python Community
