##### implement a baseline nn to check how it performs on the embeddings
## additionally it would be good to see how birdnet performs on the embeddings

import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from collections import Counter

# torch stuff to make a dataset and the baseline network 
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
class Baseline(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Baseline, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# loading the data 
def load_data(embeddings_dir, embeddings_annotation):
    annotations = pd.read_csv(embeddings_annotation)
    embeddings = []
    labels = []

    class_counts = annotations['label'].value_counts()
    valid_classes = class_counts[class_counts >= 40].index
    annotations = annotations[annotations['label'].isin(valid_classes)]
    for _, row in annotations.iterrows():
        embedding_name = row['embedding name']
        label = row['label']
        embedding_path = os.path.join(embeddings_dir, embedding_name)

        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            labels.append(label)
    
    return np.array(embeddings), np.array(labels)

def train_model(model, train_loader, val_loader,
                 criterion, optimizer, device, epochs=50):
    
    model.to(device)
    for epich in range(epochs):
        model.train()
        train_loss = 0.0

        for embeddings, label in train_loader:
            embeddings, label = embeddings.to(device), label.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)

                outputs = model(embeddings)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epich+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return model

if __name__ == '__main__':
    embeddings_dir = '../dataset/embeddings'
    embeddings_annotation = '../dataset/_embedding_annotations.csv'

    embeddings, labels = load_data(embeddings_dir, embeddings_annotation)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(embeddings, labels_encoded, 
                                                        test_size=0.3, 
                                                        random_state=42,
                                                        stratify=labels_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=0.5, 
                                                        random_state=42,
                                                        stratify=y_temp)
    
    #datasets
    train_dataset = EmbeddingDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long))
    val_dataset = EmbeddingDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))
    test_dataset = EmbeddingDataset(torch.tensor(X_test, dtype=torch.float32),
                                        torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = embeddings.shape[1]
    num_classes = len(label_encoder.classes_)
    model = Baseline(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device=DEVICE, epochs=50)

    # test set eval 

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)

            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    unique_classes = np.unique(y_true + y_pred)
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=label_encoder.inverse_transform(unique_classes), 
        labels=unique_classes
))




