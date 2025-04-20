import torch
import torch.nn as nn

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