import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset    
from utils.eval_temporal import evaluate_temporal_model
from utils.eval_temporal import softmax
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib 
import pickle 
import wandb 

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
        hidden = None
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
                try:
                    gru_out, hidden = self.gru(x)
                except Exception as e2:
                    print('Exception in GRU forward pass:', e)
                    hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
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
    
def collect(batch):
    valid_batch = []
    for window, label in batch:
        if isinstance(window, np.ndarray) and window.size > 0:
            valid_batch.append((window, label))
    
    if not valid_batch:
        print("Warning: All windows in batch were invalid")
        embedding_dim = 1024
        dummy_window = np.zeros((1, embedding_dim), dtype=np.float32)
        valid_batch = [(dummy_window, batch[0][1] if batch else 0)]
    
    valid_batch.sort(key=lambda x: len(x[0]), reverse=True)
    windows, labels = zip(*valid_batch)
    
    lengths = [max(1, len(w)) for w in windows]  
    lengths_tensor = torch.LongTensor(lengths)
    
    tensor_windows = []
    for window in windows:
        if len(window.shape) != 2:
            if len(window.shape) > 2:
                window = window.reshape(window.shape[0], -1)
            elif len(window.shape) == 1:
                window = window.reshape(1, -1)
        tensor_windows.append(torch.FloatTensor(window))
    
    padded = nn.utils.rnn.pad_sequence(tensor_windows, batch_first=True)
    
    max_len = padded.size(1)
    if torch.any(lengths_tensor > max_len):
        lengths_tensor = torch.clamp(lengths_tensor, max=max_len)
    
    lengths_tensor = torch.clamp(lengths_tensor, min=1)
    
    labels_tensor = torch.LongTensor(labels)
    return padded, lengths_tensor, labels_tensor

def sliding_window(embeddings_dir, annotations_source, window_size=5):

    if window_size % 2 == 0:
        window_size += 1
    
    radius = window_size//2

    if isinstance(annotations_source, str):
        annotations = pd.read_csv(annotations_source)
    else:
        annotations = annotations_source
    
    #parses the file so I can easily choose what will be included in the 'context'
    def parser(filename):
        is_augmented = filename.startswith('aug_')

        if is_augmented:
            clean_name = filename[4:]

        else:
            clean_name = filename

        parts = clean_name.split('_')
        recording_id = '_'.join(parts[:3])

        segment_num = 0
        subsegment_num = 0

        if len(parts) >= 5:
            try:
                segment_num = int(parts[4])
            except ValueError:
                segment_num = 0
        
        if len(parts) >= 6:
            subsegment_str = parts[5].split('.')[0]
            try:
                subsegment_num = int(subsegment_str)

            except ValueError:      
                subsegment_num = 0

        return recording_id, segment_num, subsegment_num, is_augmented
    
    parsed_info = [parser(filename) for filename in annotations['embedding name']]
    annotations['recording_id'] = [info[0] for info in parsed_info]
    annotations['segment_num'] = [info[1] for info in parsed_info]
    annotations['subsegment_num'] = [info[2] for info in parsed_info]
    #keeps track of whether the embedding is augmented or not
    annotations['is_augmented'] = [info[3] for info in parsed_info]

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
                       window_size=5, hidden_dim=256, num_layers=2, batch_size=32, 
                       epochs=50, lr=0.003, weight_decay=0.005, subset=False, return_detailed_metrics=False,
                       use_smote = True, save_resampled=True, resampled_cache_dir="../cache"
    ):

    assert model_type in ['gru', 'lstm'], "model_type must be either 'gru' or 'lstm'"

    #wandb
    run = wandb.init(
        project="bird-classification-tcm",
        name=f"{model_type}_w{window_size}_h{hidden_dim}_l{num_layers}",
        group="temporal_models",
        tags=[model_type, "augmented" if "augmented" in embeddings_dir else "regular"],
        config={
            "model_type": model_type,
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "subset": subset,
            "use_smote": use_smote,
            "epochs": epochs,
            "embeddings_type": "augmented" if "augmented" in embeddings_dir else "regular"
        }
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_annotations = pd.read_csv(annotations_path)

    if save_resampled and resampled_cache_dir:
        os.makedirs(resampled_cache_dir, exist_ok=True)
    
    #cache
    cache_identifier = f"{os.path.basename(embeddings_dir)}_{'subset' if subset else 'all'}"
    cache_file_y = os.path.join(resampled_cache_dir, f"y_train_resampled_{cache_identifier}.npy")
    cache_file_scaler = os.path.join(resampled_cache_dir, f"scaler_{cache_identifier}.joblib")
    
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
        annotations = all_annotations[all_annotations['label'].isin(selected_species)].copy()
        print(f"using subset of {len(selected_species)} species")
    
    else:
        annotations = all_annotations
        print(f"using all species: {len(annotations)}")
    
    X_windows, y_encoded, metadata, class_names = sliding_window(embeddings_dir, annotations, window_size=window_size)

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

    train_counts = {class_name: sum(y_train == i) for i, class_name in enumerate(class_names)}

    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    #had to save cache because it took too long to make all of the samples again
    pickle_file_X = os.path.join(resampled_cache_dir, f"X_train_resampled_{cache_identifier}.pkl")
    if save_resampled and os.path.exists(pickle_file_X) and os.path.exists(cache_file_y):
        print(f"Loading cached resampled data from {resampled_cache_dir}...")
        try:

            X_train = []
            with open(pickle_file_X, 'rb') as f:
                X_train = pickle.load(f)
                print(f"Loaded {len(X_train)} windows from pickle file")
            
            y_train = np.load(cache_file_y)
            scaler = joblib.load(cache_file_scaler)
            
            print(f"Loaded {len(X_train)} resampled windows and labels from cache")
            print("After SMOTE class distribution:", np.bincount(y_train))
            
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Will regenerate the resampled data...")
            use_cache = False
        else:
            use_cache = True
    else:
        use_cache = False
    
    if not use_cache and use_smote:
        print("Applying SMOTE oversampling...")
        print('extracting center embeddings...')
        center_idx = window_size // 2
        X_train_centers = np.array([
            x[min(center_idx, len(x)-1)] if len(x) > 0 else np.zeros((X_train[0][0].shape[0],)) 
            for x in tqdm(X_train, desc="Extracting center embeddings", total=len(X_train))])
        
        #storeoriginal so smote doesnt leak into the set
        original_sample_count = len(X_train_centers)
        
        #apply smote only to the training set
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train))))
        X_train_centers_resampled, y_train_resampled = smote.fit_resample(X_train_centers, y_train)
        
        print("After SMOTE class distribution:", np.bincount(y_train_resampled))
        print(f"Creating {len(X_train_centers_resampled)} samples ({len(X_train_centers_resampled) - original_sample_count} synthetic)")

        X_train_resampled = []
        
        for i in tqdm(range(len(X_train_centers_resampled)), desc="Creating resampled windows"):
            if i < original_sample_count:
                X_train_resampled.append(X_train[i])
            else:
                synthetic_point = X_train_centers_resampled[i]
                distances = np.linalg.norm(X_train_centers - synthetic_point, axis=1)
                nearest_idx = np.argsort(distances)[0]
                X_train_resampled.append(X_train[nearest_idx])
        
        # Update training data
        X_train = X_train_resampled
        y_train = y_train_resampled

        #save resampled data to cache
        if save_resampled and resampled_cache_dir:
            print(f"Saving resampled data to {resampled_cache_dir}...")
            pickle_file_X = os.path.join(resampled_cache_dir, f"X_train_resampled_{cache_identifier}.pkl")
            with open(pickle_file_X, 'wb') as f:
                pickle.dump(X_train, f)

            np.save(cache_file_y, y_train)

    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    if not use_cache:
        print("Scaling data...")
        flat_train_data = np.vstack([emb for seq in X_train for emb in seq])
        scaler = RobustScaler().fit(flat_train_data)

        if save_resampled and resampled_cache_dir:
            joblib.dump(scaler, cache_file_scaler)
            print(f"Scaler saved to {cache_file_scaler}")

    # Apply scaler to each embedding in each sequence
    X_train_scaled = [np.array([scaler.transform(emb.reshape(1, -1)).reshape(-1) for emb in seq]) for seq in X_train]
    X_val_scaled = [np.array([scaler.transform(emb.reshape(1, -1)).reshape(-1) for emb in seq]) for seq in X_val]
    X_test_scaled = [np.array([scaler.transform(emb.reshape(1, -1)).reshape(-1) for emb in seq]) for seq in X_test]

    train_dataset = TemporalDataset(X_train_scaled, y_train)
    val_dataset = TemporalDataset(X_val_scaled, y_val)
    test_dataset = TemporalDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect)

    input_dim = X_train[0].shape[1]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    #training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

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
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        #validation loop
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

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss / len(train_loader),
            "train/accuracy": train_correct / train_total,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)


        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                 f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if output_dir:
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_{model_name}.pt"))
    print('Training complete')

    if output_dir and os.path.exists(os.path.join(output_dir, f"best_{model_name}.pt")):
        model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{model_name}.pt")))

    #test loop
    print()
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []

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
            all_outputs.extend(outputs.cpu().numpy())
    
    test_acc = correct / total
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    all_outputs = np.vstack(all_outputs)
    proba_outputs = softmax(all_outputs, axis=1)

    detailed_metrics, summary_metrics = evaluate_temporal_model(
        proba_outputs, 
        all_labels, 
        class_names, 
        output_dir, 
        model_name,
        subset=subset,
        short_species=short_species if subset else None,
        medium_species=medium_species if subset else None,
        long_species=long_species if subset else None,
        train_counts=train_counts
)
    
    wandb.log({
        "test/accuracy": summary_metrics['accuracy'],
        "test/mAP": summary_metrics['mAP'],
        "test/macro_f1": summary_metrics['macro_f1'],
        "model/parameters": num_params
    })

    wandb.finish()

    if output_dir:
        torch.save(model.state_dict(), os.path.join(output_dir, f"best_{model_name}.pt"))

        metrics_df = detailed_metrics
        metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)

        with open(os.path.join(output_dir, f'{model_name}_summary.txt'), 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Window Size: {window_size}\n")
            f.write(f"Hidden Dimensions: {hidden_dim}\n")
            f.write(f"Layers: {num_layers}\n")
            f.write(f"Parameters: {num_params}\n")
            f.write(f"Accuracy: {test_acc:.4f}\n")
            f.write(f"mAP: {summary_metrics['mAP']:.4f}\n")
            f.write(f"macro F1: {summary_metrics['macro_f1']:.4f}\n")

    if return_detailed_metrics:
        return model, test_acc, detailed_metrics, val_loader
    else:
        return model, test_acc, summary_metrics, val_loader

if __name__ == "__main__":
    embeddings_dir = "../dataset/augmented_embeddings"
    annotations_path = "../dataset/_augmented_embedding_annotations.csv"
    output_dir = "../results/gru_results"
    cache_dir = "../cached_data"

    aug_gru_model, aug_gru_acc, aug_gru_metrics = train_model(
        model_type='gru',
        embeddings_dir=embeddings_dir,
        annotations_path=annotations_path,
        output_dir=output_dir,
        window_size= 5,
        hidden_dim=32,
        num_layers=1,        
        subset=True,
        return_detailed_metrics=True,
        epochs = 50,
        use_smote = True,
        save_resampled=True,
        resampled_cache_dir=cache_dir
    )
