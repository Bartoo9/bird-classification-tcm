import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset    

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim = 512, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  

    def forward(self, x, lengths = None):

        batch_size = x.size(0)
        if lengths is not None:

            if torch.any(lengths <= 0):
                print('zero length batch in forward pass')
                lengths = torch.clamp(lengths, min=1)

            max_length = x.size(1)

            if torch.any(lengths > max_length):
                print('some lengths exceeded padded size in forward pass')
                lengths = torch.clamp(lengths, max=max_length)

            try:
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                            batch_first=True, enforce_sorted=False)
                gru_out, hidden = self.gru(packed)
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
            
            except RuntimeError as e:
                print('RuntimeError in GRU forward pass:', e)
                print('x shape:', x.shape)
                print('lengths:', lengths)
                
        
        else:
            gru_out, hidden = self.gru(x)

        last_hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)[-1]
        last_hidden = last_hidden.permute(1, 0, 2).reshape(batch_size, -1)
        logits = self.classifier(last_hidden)

        return logits

class TemporalDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]
    
# def collect(batch):
#     batch = [(w, l) for w, l in batch if len(w) > 0]
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     windows, labels = zip(*batch)
#     lengths = torch.LongTensor([len(w) for w in windows])
    
#     tensor_windows = []
#     for window in windows:
#         if len(window.shape) != 2:
#             print('window shape:', window.shape)

#             if len(window.shape) > 2:
#                 window = window.reshape(window.shape[0], -1)
#         tensor_windows.append(torch.FloatTensor(window))
        

#     padded = nn.utils.rnn.pad_sequence(
#         tensor_windows, 
#         batch_first=True)
    
#     max_len = padded.size(1)
#     if torch.any(lengths > max_len):
#         print('some lengths exceeded padded size')
#         lengths = torch.clamp(lengths, max=max_len)

#     labels_tensor = torch.LongTensor(labels)

#     return padded, labels_tensor, lengths

def collect(batch):
    # First, filter out any empty or invalid windows
    valid_batch = []
    for window, label in batch:
        if isinstance(window, np.ndarray) and window.size > 0:
            valid_batch.append((window, label))
    
    # If no valid windows remain, create a minimal valid batch
    if not valid_batch:
        print("Warning: All windows in batch were invalid")
        # Create a dummy window with minimal valid dimensions
        dummy_window = np.zeros((1, batch[0][0].shape[1] if batch and hasattr(batch[0][0], 'shape') else 10))
        valid_batch = [(dummy_window, batch[0][1] if batch else 0)]
    
    # Sort by length
    valid_batch.sort(key=lambda x: len(x[0]), reverse=True)
    windows, labels = zip(*valid_batch)
    
    # Get lengths and verify they're all positive
    lengths = [max(1, len(w)) for w in windows]  # Force minimum length 1
    lengths_tensor = torch.LongTensor(lengths)
    
    # Create tensor windows with proper shape checking
    tensor_windows = []
    for window in windows:
        if len(window.shape) != 2:
            if len(window.shape) > 2:
                # Reshape to 2D (sequence length, features)
                window = window.reshape(window.shape[0], -1)
            elif len(window.shape) == 1:
                # Expand to 2D (1 sequence element, features)
                window = window.reshape(1, -1)
        tensor_windows.append(torch.FloatTensor(window))
    
    # Pad sequences
    padded = nn.utils.rnn.pad_sequence(tensor_windows, batch_first=True)
    
    # Double-check that lengths don't exceed padded size
    max_len = padded.size(1)
    if torch.any(lengths_tensor > max_len):
        lengths_tensor = torch.clamp(lengths_tensor, max=max_len)
    
    # Final sanity check - no zero lengths
    lengths_tensor = torch.clamp(lengths_tensor, min=1)
    
    labels_tensor = torch.LongTensor(labels)
    return padded, lengths_tensor, labels_tensor

def sliding_window(embeddings_dir, annotations_path, window_size=5):

    if window_size % 2 == 0:
        window_size += 1
    
    radius = window_size//2

    annotations = pd.read_csv(annotations_path)
    
    #parses the file so I can easily choose what will be included in the 'context'
    def parser(filename):
        parts = filename.split('_')
        recording_id = '_'.join(parts[:3])
        segment_num = int(parts[4])
        subsegment_num = int(parts[5].split('.')[0])

        return recording_id, segment_num, subsegment_num
    
    parsed_info = [parser(filename) for filename in annotations['embedding name']]
    annotations['recording_id'] = [info[0] for info in parsed_info]
    annotations['segment_num'] = [info[1] for info in parsed_info]
    annotations['subsegment_num'] = [info[2] for info in parsed_info]

    recording_groups = annotations.groupby('recording_id')

    X_windows = []
    y_targets = []
    metadata = []

    for recording_id, group in recording_groups:
        sorted_group = group.sort_values(by=['segment_num', 'subsegment_num'])

        recording_embeddings = []
        recording_labels = []
        embedding_names = []

        for _, row in sorted_group.iterrows():
            embedding_path = os.path.join(embeddings_dir, row['embedding name'])
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                recording_embeddings.append(embedding)
                recording_labels.append(row['label'])
                embedding_names.append(row['embedding name'])

        for i in range(len(recording_embeddings)):
            start_idx = max(0, i - radius)
            end_idx = min(len(recording_embeddings), i + radius + 1)

            if end_idx <= start_idx:
                print('empty window')
                continue
            
            window = np.array(recording_embeddings[start_idx:end_idx])

            if len(window) == 0:
                print('zero length window found at', recording_id)
                continue

            center_label = recording_labels[i]

            X_windows.append(window)
            y_targets.append(center_label)
            metadata.append({
                'recording_id': recording_id,
                'center_label': embedding_names[i],
                'segment_num': sorted_group.iloc[i]['segment_num'],
                'subsegment_num': sorted_group.iloc[i]['subsegment_num'],
                'window_start': start_idx,
                'window_end': end_idx
            })
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_targets)

    print(f'created {len(X_windows)} windows from {len(recording_groups)} recordings')

    return X_windows, y_encoded, metadata, label_encoder.classes_



def train_model(model_type, embeddings_dir, annotations_path, output_dir=None, 
                       window_size=5, hidden_dim=512, num_layers=2, batch_size=32, 
                       epochs=100, lr=0.001, weight_decay=0.01, subset=False):
    assert model_type in ['gru', 'lstm'], "model_type must be either 'gru' or 'lstm'"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_annotations = pd.read_csv(annotations_path)

    if subset:
        short_species = [
            'Emberiza calandra', 'Cyanistes caeruleus', 'Regulus regulus',
            'Certhia brachydactyla', 'Emberiza cirlus'
        ]

        medium_species = [
            'Fringilla coelebs', 'Turdus merula', 'Sylvia atricapilla',
            'Erithacus rubecula', 'Luscinia megarhynchos', 'Parus major',
            'Phylloscopus collybita', 'Turdus philomelos'
        ]

        long_species = [
            'Periparus ater', 'Alauda arvensis', 'Cuculus canorus',
            'Acrocephalus scirpaceus', 'Acrocephalus arundinaceus', 
            'Columba palumbus', 'Spinus spinus'
        ]
        
        selected_species = short_species + medium_species + long_species
        annotations = all_annotations[all_annotations['label'].isin(selected_species)]
        print(f"using subset of {len(selected_species)} species")
    
    else:
        annotations = all_annotations
        print(f"using all species: {len(annotations)}")
    
    X_windows, y_encoded, metadata, class_names = sliding_window(embeddings_dir, annotations_path, window_size=window_size)

    unique_recordings = list(set([m['recording_id'] for m in metadata]))
    train_val_recs, test_recs = train_test_split(unique_recordings, test_size=0.1, random_state=1)
    train_recs, val_recs = train_test_split(train_val_recs, test_size=0.1, random_state=1)

    train_mask = np.array([m['recording_id'] in train_recs for m in metadata])
    val_mask = np.array([m['recording_id'] in val_recs for m in metadata])
    test_mask = np.array([m['recording_id'] in test_recs for m in metadata])

    X_train = [X_windows[i] for i in range(len(X_windows)) if train_mask[i]]
    y_train = y_encoded[train_mask]
    X_val = [X_windows[i] for i in range(len(X_windows)) if val_mask[i]]
    y_val = y_encoded[val_mask]
    X_test = [X_windows[i] for i in range(len(X_windows)) if test_mask[i]]
    y_test = y_encoded[test_mask]

    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    train_dataset = TemporalDataset(X_train, y_train)
    val_dataset = TemporalDataset(X_val, y_val)
    test_dataset = TemporalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect)

    input_dim = X_train[0].shape[1]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if model_type == 'gru':
    model = GRUClassifier(
            input_dim = input_dim,
             num_classes= num_classes, 
             hidden_dim=hidden_dim, 
             num_layers=num_layers).to(device)
        
    model_name = f'gru_model_w{window_size}_h{hidden_dim}_l{num_layers}'
    
    print(f'training {model_name}')

    #imbalance 
    class_counts = np.bincount(y_train)
    class_weights = np.sqrt(class_counts.max() / class_counts)
    class_weights = class_weights / np.mean(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, lengths, y in train_loader:

            if x.size(0) == 0:
                print('empty batch')
                continue

            if torch.any(lengths <= 0):
                print('zero length batch')
                lengths = torch.clamp(lengths, min=1)

            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        train_acc = correct / total
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                outputs = model(x, lengths)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
        val_acc = correct / total
        val_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                 f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Val Acc: {val_acc:.4f}")
            
        if output_dir and os.path.exists(os.path.join(output_dir, f"best_{model_name}.pt")):
            model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{model_name}.pt")))

        print()
        model.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, lengths, y in test_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                outputs = model(x, lengths)
                loss = criterion(outputs, y)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        test_acc = correct / total
        test_loss /= len(test_loader)


if __name__ == "__main__":
    # Example usage
    embeddings_dir = "../dataset/embeddings"
    annotations_path = "../dataset/_embedding_annotations.csv"
    output_dir = "../results/gru"
    
    # Run GRU model
    gru_model, gru_acc, _ = train_model(
        model_type='gru',
        embeddings_dir=embeddings_dir,
        annotations_path=annotations_path,
        output_dir=output_dir,
        window_size=5,
        hidden_dim=512,
        num_layers=2,        subset=True
    )
    
    # Run transformer model
    # transformer_model, transformer_acc, _ = train_model(
    #     model_type='transformer',
    #     embeddings_dir=embeddings_dir,
    #     annotations_path=annotations_path,
    #     output_dir=os.path.join(output_dir, "transformer"),
    #     window_size=5,
    #     hidden_dim=512,
    #     num_layers=4,  # Transformers typically use more layers
    #     subset=True
    # )
    
    print(f"\nComparison:")
    print(f"GRU Accuracy: {gru_acc:.4f}")
    # print(f"Transformer Accuracy: {transformer_acc:.4f}")
