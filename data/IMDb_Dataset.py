import os
import torch
from torch.utils.data import Dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class IMDbReviewDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        self.reviews = []
        self.labels = []
        self.ratings = []
        self.transform = transform
        
        # Initialize the vectorizer (you can adjust parameters like max_features)
        self.vectorizer = TfidfVectorizer(max_features=10000)

        # Load positive and negative reviews from the directories
        self._load_data(pos_dir, 1)
        self._load_data(neg_dir, 0)

        # Fit the vectorizer on the reviews (it's important to fit the vectorizer on the entire dataset)
        all_reviews = self.reviews
        self.vectorizer.fit(all_reviews)

    def _load_data(self, directory, label):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                # Extract rating from filename, which is before the '.txt'
                match = re.match(r"\d+_(\d+)\.txt", filename)
                if match:
                    rating = int(match.group(1))  # Rating is extracted from the filename
                    self.reviews.append(text)
                    self.labels.append(label)
                    self.ratings.append(rating)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        rating = self.ratings[idx]
        
        # Convert the review text to a numeric feature vector using TF-IDF
        review_vector = self.vectorizer.transform([review]).toarray()  # Convert to array and then tensor
        review_tensor = torch.tensor(review_vector, dtype=torch.float32)

        return review_tensor.squeeze(0), label, rating
