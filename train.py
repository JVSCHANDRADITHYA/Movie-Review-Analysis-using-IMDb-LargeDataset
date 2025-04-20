import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

train_pos_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\train\pos'  # Replace with your 'pos' train folder path
train_neg_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\train\neg'  # Replace with your 'neg' train folder path
test_pos_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\pos'  # Replace with your 'pos' test folder path
test_neg_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\neg'  # Replace with your 'neg' test folder path

# Function to load reviews from the directories
def load_reviews_from_dir(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            # Extract rating from filename, which is before the '.txt'
            match = re.match(r"\d+_(\d+)\.txt", filename)
            if match:
                rating = int(match.group(1))  # Rating is extracted from the filename
                data.append({
                    'review': text,
                    'label': label,
                    'rating': rating
                })
    return data

# Load training data (positive and negative)
train_pos_reviews = load_reviews_from_dir(train_pos_dir, 1)
train_neg_reviews = load_reviews_from_dir(train_neg_dir, 0)

# Load testing data (positive and negative)
test_pos_reviews = load_reviews_from_dir(test_pos_dir, 1)
test_neg_reviews = load_reviews_from_dir(test_neg_dir, 0)

# Combine train data
train_reviews = pd.DataFrame(train_pos_reviews + train_neg_reviews)
# Combine test data
test_reviews = pd.DataFrame(test_pos_reviews + test_neg_reviews)

# Shuffle the training data
train_reviews = train_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

# Shuffle the testing data
test_reviews = test_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and targets
X_train = train_reviews['review']  # Reviews text
y_train_cls = train_reviews['label']    # 0 or 1 for sentiment
y_train_reg = train_reviews['rating']  # Ratings (1-10)

X_test = test_reviews['review']  # Reviews text
y_test_cls = test_reviews['label']    # 0 or 1 for sentiment
y_test_reg = test_reviews['rating']  # Ratings (1-10)

# Classification model pipeline
clf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(max_iter=200))
])

# Train classification model
clf_pipeline.fit(X_train, y_train_cls)

# Regression model pipeline
reg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('reg', Ridge())
])

# Train regression model
reg_pipeline.fit(X_train, y_train_reg)

# Save the trained models
joblib.dump(clf_pipeline, 'imdb_sentiment_classifier.pkl')
joblib.dump(reg_pipeline, 'imdb_rating_regressor.pkl')

# Evaluate models on test data
y_pred_cls = clf_pipeline.predict(X_test)
y_pred_reg = reg_pipeline.predict(X_test)

# Accuracy for classification
cls_acc = accuracy_score(y_test_cls, y_pred_cls)

# Mean Squared Error for regression
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

# Output the evaluation metrics
print(f"Classification Accuracy: {cls_acc:.4f}")
print(f"Regression MSE: {reg_mse:.4f}")
