import torch
from torch.utils.data import DataLoader
from data.IMDb_Dataset import IMDbReviewDataset
from model.sentiment_rating import SentimentRatingModel  # Assuming model.py is in the same directory
import numpy as np
from sklearn.metrics import mean_squared_error

# Set the paths to your test data (same as the training script)
test_pos_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\pos'
test_neg_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\neg'

# Initialize the test dataset and DataLoader
test_dataset = IMDbReviewDataset(test_pos_dir, test_neg_dir, transform=None)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_dim = 10000  # Adjust this according to the TF-IDF vectorizer's max_features
model = SentimentRatingModel(input_dim)

# Load the trained model state dict
model.load_state_dict(torch.load('sentiment_rating_model.pth'))
model.eval()  # Set the model to evaluation mode

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to evaluate the model
def evaluate_model():
    all_labels = []
    all_ratings = []
    all_sentiment_preds = []
    all_rating_preds = []

    with torch.no_grad():  # Disable gradient computation
        for reviews, labels, ratings in test_loader:
            reviews = reviews.to(device)
            labels = labels.to(device).float()
            ratings = ratings.to(device).float()

            # Get predictions
            rating_preds, sentiment_preds = model(reviews)

            # Store the results
            all_labels.extend(labels.cpu().numpy())
            all_ratings.extend(ratings.cpu().numpy())
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_rating_preds.extend(rating_preds.cpu().numpy())

    # Evaluate performance (you can add accuracy or any other metric)
    # Convert predictions to binary labels for sentiment classification
    sentiment_preds_binary = (np.array(all_sentiment_preds) > 0.5).astype(int)

    # Calculate Accuracy for Sentiment Classification
    accuracy = (sentiment_preds_binary == np.array(all_labels)).mean()
    print(f"Sentiment Classification Accuracy: {accuracy:.4f}")

    # Calculate Mean Squared Error (MSE) for Rating Prediction
    mse = mean_squared_error(all_ratings, all_rating_preds)
    print(f"Mean Squared Error for Rating Prediction: {mse:.4f}")

# Run the evaluation
evaluate_model()
