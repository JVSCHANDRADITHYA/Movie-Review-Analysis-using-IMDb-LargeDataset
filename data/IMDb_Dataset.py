import os
import torch
from torch.utils.data import Dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

class IMDbReviewDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        self.reviews = []
        self.labels = []
        self.ratings = []
        self.transform = transform
        
        self._load_data(pos_dir, 1)
        self._load_data(neg_dir, 0)

    def _load_data(self, directory, label):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                match = re.match(r"\d+_(\d+)\.txt", filename)
                if match:
                    rating = int(match.group(1)) 
                    self.reviews.append(text)
                    self.labels.append(label)
                    self.ratings.append(rating)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        rating = self.ratings[idx]
        
        if self.transform:
            review = self.transform(review)

        return review, label, rating
