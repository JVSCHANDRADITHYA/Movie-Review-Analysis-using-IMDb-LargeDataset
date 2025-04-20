import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from data.IMDb_Dataset import IMDbReviewDataset
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

# Set the paths to your train/pos and train/neg directories
train_pos_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\train\pos'  # Replace with your 'pos' train folder path
train_neg_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\train\neg'  # Replace with your 'neg' train folder path
test_pos_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\pos'  # Replace with your 'pos' test folder path
test_neg_dir = r'F:\Movie-Review-Analysis-using-IMDb-LargeDataset\IMDb_dataset\test\neg'  # Replace with your 'neg' test folder path

# Define a simple feedforward neural network for sentiment classification and regression
class SentimentRatingModel(nn.Module):
    def __init__(self, input_dim):
        super(SentimentRatingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # For regression output (rating prediction)
        self.fc4 = nn.Linear(64, 1)  # For classification output (binary sentiment prediction)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        rating_output = self.fc3(x)  # For regression
        sentiment_output = torch.sigmoid(self.fc4(x))  # For classification (0 or 1)
        return rating_output, sentiment_output

# Load the dataset
train_dataset = IMDbReviewDataset(train_pos_dir, train_neg_dir, transform=None)
test_dataset = IMDbReviewDataset(test_pos_dir, test_neg_dir, transform=None)

# Set up DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_dim = 10000  # This can be adjusted based on your TF-IDF vectorizer max_features
model = SentimentRatingModel(input_dim)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss functions and optimizer
criterion_cls = nn.BCEWithLogitsLoss()  # For binary classification
criterion_reg = nn.MSELoss()  # For regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    # In your training loop, the review data is now a tensor
    for reviews, labels, ratings in train_loader:
        reviews = reviews.to(device)
        labels = labels.to(device).float()
        ratings = ratings.to(device).float()

        # Get the predictions
        rating_preds, sentiment_preds = model(reviews)

        # Compute the losses
        loss_cls = criterion_cls(sentiment_preds.squeeze(), labels)
        loss_reg = criterion_reg(rating_preds.squeeze(), ratings)
        loss = loss_cls + loss_reg

        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Save the trained model
# Save the trained model using torch.save()
torch.save(model.state_dict(), 'sentiment_rating_model.pth')


