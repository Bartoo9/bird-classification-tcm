import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset    
from utils.eval_temporal import evaluate_temporal_model
from sklearn.preprocessing import RobustScaler 
import wandb 
from utils.window_sizes import analyze_window_sizes
import torch.nn.functional as F
from utils.stratified_sampling import stratified_split_by_recording

#gru class - kept as simple as possible
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim = 512, num_layers=2, dropout=0.3, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size=input_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
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

        if self.bidirectional:
            foward_hidden = hidden[0:hidden.size(0):2]
            backward_hidden = hidden[1:hidden.size(0):2]
            last_hidden = torch.cat((foward_hidden[-1], backward_hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)

        return logits

#temporal dataset object
class TemporalDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

#collate function to collect all of the windows in a batch
def collect(batch):
    valid_batch = []
    window_sizes = []
    labels = []

    for window, label in batch:
        if isinstance(window, np.ndarray) and window.size > 0:
            valid_batch.append((window, label))
            window_sizes.append(len(window))
            labels.append(label)
    
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
    
    labels_tensor = torch.FloatTensor(np.vstack(labels))

    return padded, lengths_tensor, labels_tensor

#generates the sliding windows based on the given size and directionality
def sliding_window(embeddings_dir, annotations_source, window_size=5, ctx_type='bidirectional', subset=True):

    if isinstance(annotations_source, str):
        print(f"Loading annotations from file: {annotations_source}")
        if not os.path.exists(annotations_source):
            print(f"ERROR: Annotations file not found: {annotations_source}")
            return [], [], [], []
        annotations = pd.read_csv(annotations_source)
    else:
        annotations = annotations_source.copy()
    
    print(f"Initial annotations shape: {annotations.shape}")
    
    required_columns = ['segment_name', 'species', 'embedding_name']
    missing_columns = [col for col in required_columns if col not in annotations.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns in annotations: {missing_columns}")
        print(f"Available columns: {annotations.columns.tolist()}")
        return [], [], [], []
    
    if window_size % 2 == 0 and ctx_type == 'bidirectional':
        window_size += 1
    
    if ctx_type == 'bidirectional':
        past_ctx = window_size // 2
        future_ctx = window_size // 2
    else:
        past_ctx = window_size - 1
        future_ctx = 0

    if subset:
        short_species = [
            "Cyanistes caeruleus_Eurasian Blue Tit",
            "Regulus regulus_Goldcrest",
            "Cisticola juncidis_Landeryklopkloppie",
            "Parus major_Great Tit",
            "Erithacus rubecula_European Robin",
            "Emberiza calandra_Corn Bunting",
            "Troglodytes troglodytes_Eurasian Wren",
            "Phylloscopus trochilus_Hofsanger",
            "Curruca melanocephala_Sardinian Warbler"
        ]

        medium_species = [
            "Fringilla coelebs_Gryskoppie", 
            "Turdus merula_Eurasian Blackbird",
            "Sylvia atricapilla_Swartkroonsanger",
            "Luscinia megarhynchos_Common Nightingale",
            "Phylloscopus collybita_Common Chiffchaff",
            "Turdus philomelos_Song Thrush",
            "Periparus ater_Coal Tit",
            "Phylloscopus sibilatrix_Wood Warbler",
            "Galerida theklae_Thekla's Lark",
            "Lullula arborea_Wood Lark"
        ]

        long_species = [
            "Alauda arvensis_Eurasian Skylark",
            "Cuculus canorus_Europese Koekoek",
            "Acrocephalus scirpaceus_Common Reed Warbler",
            "Acrocephalus arundinaceus_Grootrietsanger",
            "Columba palumbus_Common Wood-Pigeon"
        ]
        
    if subset:
        selected_species = short_species + medium_species + long_species + ["Noise_Noise"]
        print(f"Using subset of {len(selected_species)} species")
        
        import ast
        
        def parse_species(species_entry):
            if isinstance(species_entry, str):
                if species_entry.startswith('[') and species_entry.endswith(']'):
                    try:
                        return ast.literal_eval(species_entry)
                    except:
                        return [species_entry]
                return [species_entry]
            elif isinstance(species_entry, list):
                return species_entry
            else:
                return [str(species_entry)]
        
        print(f"Annotations before filtering: {len(annotations)}")
        annotations['species_list'] = annotations['species'].apply(parse_species)

        annotations = annotations[annotations['species_list'].apply(
            lambda species_list: all(species in selected_species for species in species_list)
        )]
        print(f"Annotations after strict filtering: {len(annotations)}")
        
        if len(annotations) == 0:
            print("WARNING: All annotations were filtered out! Check species names.")
            print(f"Selected species (first 3): {selected_species[:3]}")
            sample_species = annotations_source['species'].iloc[:5].tolist() if isinstance(annotations_source, pd.DataFrame) else "N/A"
            print(f"Sample species in data: {sample_species}")
            return [], [], [], []

    #parser of the files based on the naming of the embeddings
    def parser(filename):
        if filename.endswith('.npy') or filename.endswith('.wav'):
            filename = os.path.splitext(filename)[0]

        clean_name = filename
        parts = clean_name.split('_')
        
        recording_id = '_'.join(parts[:2]) if len(parts) >= 2 else filename
        
        part = "unknown"
        segment_num = 0
        subsegment_num = 0
        
        if len(parts) > 6:
            part = parts[6]

            part_ranks = {
                'start': 1,
                'middle': 2,
                'end': 3,
                'background': 1
            }
            segment_num = part_ranks.get(part, 0)

        if len(parts) > 7:
            try:
                subsegment_num = int(parts[7])
            except ValueError:
                subsegment_num = 0

        return recording_id, segment_num, part, subsegment_num
    
    parsed_info = [parser(filename) for filename in annotations['segment_name']]
    annotations['recording_id'] = [info[0] for info in parsed_info]
    annotations['segment_num'] = [info[1] for info in parsed_info]
    annotations['position'] = [info[2] for info in parsed_info] 
    annotations['subsegment_num'] = [info[3] for info in parsed_info]
    recording_groups = annotations.groupby('recording_id')

    X_windows = []
    y_targets = []
    metadata = []

    all_species_set = set()
    for species_list in annotations['species_list']:
        if isinstance(species_list, list):
            all_species_set.update(species_list)
        else:
            all_species_set.add(species_list)

    all_species = sorted(all_species_set)
    species_to_idx = {species: idx for idx, species in enumerate(all_species)}
    print(f"Created index for {len(all_species)} unique species")

    num_classes = len(all_species)

    for recording_id, group in recording_groups:
        sorted_group = group.sort_values(by=['segment_num'])

        recording_embeddings = []
        recording_timestamps = []
        recording_labels = []
        embedding_names = []
        positions = []

        for _, row in sorted_group.iterrows():
            embedding_path = os.path.join(embeddings_dir, row['embedding_name'])
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                recording_embeddings.append(embedding)
                recording_timestamps.append(row['segment_num'])
                recording_labels.append(row['species_list'] if 'species_list' in row else row['species'])
                embedding_names.append(row['embedding_name'])
                positions.append(row['position'])

        for i in range(len(recording_embeddings)):
            start_idx = max(0, i - past_ctx)
            end_idx = min(len(recording_embeddings), i + future_ctx + 1)

            if end_idx <= start_idx:
                print('empty window')
                continue
            
            window = np.array(recording_embeddings[start_idx:end_idx])

            if len(window) == 0:
                print('zero length window found at', recording_id)
                continue
            
            window_species = set()
            for species_list in recording_labels[start_idx:end_idx]:
                if isinstance(species_list, list):
                    window_species.update(species_list)
                else:
                    window_species.add(species_list)

            target = np.zeros(num_classes)

            for species in window_species:
                if species in species_to_idx:
                    target[species_to_idx[species]] = 1

            X_windows.append(window)
            y_targets.append(target)
            metadata.append({
                'recording_id': recording_id,
                'center_label': embedding_names[i],
                'position': positions[i],
                'window_start': start_idx,
                'window_end': end_idx,
                'window_size': window_size,
                'ctx type': ctx_type,
                'num_species': len(window_species)
            })
    
    y_targets = np.array(y_targets)
    y_targets_array = np.array(y_targets)

    if len(y_targets_array.shape) != 2:
        print(f"Warning: y_targets has unexpected shape {y_targets_array.shape}")
        if len(y_targets) > 0:
            if num_classes > 0:
                y_targets_array = np.vstack([np.array(t).reshape(1, -1) for t in y_targets])
            else:
                y_targets_array = np.zeros((len(y_targets), 1))
        else:
            y_targets_array = np.zeros((0, max(1, num_classes)))
    
    if len(y_targets_array) > 0:
        num_single_label = sum(np.sum(y_targets_array, axis=1) == 1)
        num_multi_label = sum(np.sum(y_targets_array, axis=1) > 1)
        avg_labels = np.mean(np.sum(y_targets_array, axis=1))
    else:
        num_single_label = num_multi_label = 0
        avg_labels = 0


    print(f'Created {len(X_windows)} windows from {len(recording_groups)} recordings')
    
    
    print(f'Multi-label statistics:')
    print(f'  - Windows with single species: {num_single_label} ({num_single_label/len(X_windows)*100:.1f}%)')
    print(f'  - Windows with multiple species: {num_multi_label} ({num_multi_label/len(X_windows)*100:.1f}%)')
    print(f'  - Average species per window: {avg_labels:.2f}')

    return X_windows, y_targets, metadata, all_species

#train model fucntion including subsetting, train val and test loops
def train_model(model_type, embeddings_dir, annotations_path, output_dir=None, 
                       window_size=5, hidden_dim=256, num_layers=2, batch_size=32, 
                       epochs=50, lr=0.001, weight_decay=0.01, subset=False, return_detailed_metrics=False,
                       ctx_type='bidirectional', downsample=False, threshold=1500, dropout=0.3, sigmoid=0.4,
                       use_wandb=True
    ):

    short_species = [
        "Cyanistes caeruleus_Eurasian Blue Tit",
        "Regulus regulus_Goldcrest",
        "Cisticola juncidis_Landeryklopkloppie",
        "Parus major_Great Tit",
        "Erithacus rubecula_European Robin",
        "Emberiza calandra_Corn Bunting",
        "Troglodytes troglodytes_Eurasian Wren",
        "Phylloscopus trochilus_Hofsanger",
        "Curruca melanocephala_Sardinian Warbler"
    ]

    medium_species = [
        "Fringilla coelebs_Gryskoppie", 
        "Turdus merula_Eurasian Blackbird",
        "Sylvia atricapilla_Swartkroonsanger",
        "Luscinia megarhynchos_Common Nightingale",
        "Phylloscopus collybita_Common Chiffchaff",
        "Turdus philomelos_Song Thrush",
        "Periparus ater_Coal Tit",
        "Phylloscopus sibilatrix_Wood Warbler",
        "Galerida theklae_Thekla's Lark",
        "Lullula arborea_Wood Lark"
    ]

    long_species = [
        "Alauda arvensis_Eurasian Skylark",
        "Cuculus canorus_Europese Koekoek",
        "Acrocephalus scirpaceus_Common Reed Warbler",
        "Acrocephalus arundinaceus_Grootrietsanger",
        "Columba palumbus_Common Wood-Pigeon"
    ]

    assert model_type in ['gru']

    #wandb
    if use_wandb:
        try:
            if wandb.run is None:
                wandb.init(project="bird-classification", 
                          name=f"{model_type}_w{window_size}_h{hidden_dim}_l{num_layers}")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_annotations = pd.read_csv(annotations_path)
    

    X_windows, y_encoded, metadata, class_names = sliding_window(embeddings_dir, all_annotations,
                                                                  window_size=window_size, ctx_type=ctx_type,
                                                                  subset=subset)

    X_array = np.array(X_windows, dtype=object)
    y_array = np.array(y_encoded)


    print("\nSplitting data using multi-label stratified split...")
    try:
        #here the stratified sampling is used
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_by_recording(
            X_array, y_array, test_size=0.1, val_size=0.1, random_state=42)
        
        print(f"Split complete: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    except Exception as e:
        X_train, X_test, y_train, y_test = train_test_split(X_windows, y_encoded, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42) 

    print("\nAnalyzing window sizes for the training set:")
    window_stats = analyze_window_sizes(X_windows, y_encoded, class_names, output_dir)
    if use_wandb and wandb.run is not None:
        wandb.log({"window_size_stats": wandb.Table(
            columns=["Species", "Count", "Mean Size", "Min Size", "Max Size"],
            data=[[row['species'], row['count'], row['avg_size'], row['min_size'], row['max_size']] 
                for _, row in window_stats.iterrows()]
        )})

    #class distributions
    class_distribution = pd.DataFrame({
        'class': class_names,
        'train': [np.sum(y_train[:, i]) for i in range(len(class_names))],
        'val': [np.sum(y_val[:, i]) for i in range(len(class_names))],
        'test': [np.sum(y_test[:, i]) for i in range(len(class_names))]
    })

    class_distribution['total'] = class_distribution['train'] + class_distribution['val'] + class_distribution['test']
    class_distribution['train_pct'] = (class_distribution['train'] / class_distribution['train'].sum()) * 100
    class_distribution['val_pct'] = (class_distribution['val'] / class_distribution['val'].sum()) * 100
    class_distribution['test_pct'] = (class_distribution['test'] / class_distribution['test'].sum()) * 100

    print("\nClass distribution after stratified split:")
    print(class_distribution)

    if use_wandb and wandb.run is not None:
        wandb.log({
            "class_distribution": wandb.Table(
                dataframe=class_distribution
            )
        })

    if downsample:
        print(f"\nDownsampling majority classes (threshold: {threshold} samples)")
        class_counts = np.sum(y_train, axis=0)
        print("Original class distribution:", class_counts)
        class_indices = [np.where(y_train == i)[0] for i in range(len(class_counts))]
        X_train_downsampled = []
        y_train_downsampled = []

        for class_idx, indices in enumerate(class_indices):
            count = len(indices)
            if count > threshold:
                np.random.seed(4)
                indices = np.random.choice(indices, size=threshold, replace=False)
                print(f"Class {class_names[class_idx]}: {count} -> {threshold} samples")
            else:
                indices = indices
            
            for idx in indices:
                X_train_downsampled.append(X_train[idx])
                y_train_downsampled.append(y_train[idx])
    
        X_train = X_train_downsampled
        y_train = np.array(y_train_downsampled)

        new_class_counts = np.sum(y_train, axis=0)
        print("New class distribution:", new_class_counts)
    if isinstance(X_train[0], np.ndarray):
        flat_train_data = np.vstack([emb for seq in X_train for emb in seq])
    else:
        flat_train_data = np.vstack([emb for seq in X_train for emb in seq])

    scaler = RobustScaler().fit(flat_train_data)

    #scaling the values
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

    #model
    model = GRUClassifier(
            input_dim = input_dim,
             num_classes= num_classes, 
             hidden_dim=hidden_dim, 
             num_layers=num_layers,
             bidirectional=(ctx_type == 'bidirectional')).to(device)
        
    model_name = f'gru_model_w{window_size}_h{hidden_dim}_l{num_layers}'
    
    print(f'training {model_name}')

    class_counts = np.sum(y_train, axis=0)
    pos_samples = class_counts
    neg_samples = len(y_train) - class_counts
    pos_weight = neg_samples / np.maximum(pos_samples, 1) 

    #cap large weights due to the imbalanced nature of the dataset
    pos_weight = np.minimum(pos_weight, 100.0)

    class_weights_tensor = torch.FloatTensor(pos_weight).to(device)
    print(f"Class weight range: [{np.min(pos_weight):.2f}, {np.max(pos_weight):.2f}]")

    #focal loss
    from utils.focal_loss import FocalLoss
    criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean', pos_weight=class_weights_tensor)

    # class_weights_tensor = torch.FloatTensor(class_weights).to(device) 
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',          
        factor=0.5,          
        patience=5,          
        verbose=True,
        min_lr=1e-6
    )
    
    best_val_acc = 0.0

    #training loop
    for epoch in range(epochs):
        if epoch == 0:
            #some checks for the first epoch
            print("\nInitial model statistics:")
            first_batch = next(iter(train_loader))
            first_x, first_lengths, first_y = [item.to(device) for item in first_batch]
            model.eval()
            with torch.no_grad():
                outputs = model(first_x, first_lengths)
                probs = F.sigmoid(outputs)
            print(f"- Output logits range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"- Output probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            print(f"- Unique predicted classes: {len(torch.unique(outputs.argmax(dim=1)))}")
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
            train_preds = (torch.sigmoid(outputs) > sigmoid).int()
            TP = ((train_preds == 1) & (y == 1)).sum(dim=1).float()
            TN = ((train_preds == 0) & (y == 0)).sum(dim=1).float()
            total_correct = TP + TN
            total_per_sample = y.size(1) 
            sample_accuracies = total_correct / total_per_sample
            train_correct += torch.sum(sample_accuracies).item()
            train_total += y.size(0)

        train_acc = float(train_correct) / train_total
        train_loss /= len(train_loader)

        if epoch % 10 == 0 or epoch == epochs-1:
            all_train_preds = []
            model.eval()
            with torch.no_grad():
                for x, lengths, y in train_loader:
                    x, lengths = x.to(device), lengths.to(device)
                    outputs = model(x, lengths)
                    _, predicted = outputs.max(1)
                    all_train_preds.extend(predicted.cpu().numpy())
            all_train_preds_array = np.vstack(all_train_preds)
            classes_with_positive_preds = np.sum(np.sum(all_train_preds_array, axis=0) > 0)
            print(f"- Training: predicting {classes_with_positive_preds} unique classes out of {num_classes}")
            model.train()

        #validation loop
        model.eval()
        val_loss = 0.0
        correct = 0.0
        val_total = 0.0

        all_val_preds = []
        all_val_outputs = []

        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                outputs = model(x, lengths)
                targets = y.float().to(device)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_preds = (torch.sigmoid(outputs) > sigmoid).int()
                all_val_outputs.append(outputs.cpu().numpy())
                preds_per_sample = torch.sum(val_preds, dim=1)
                all_val_preds.append(val_preds.cpu().numpy())
                TP = ((val_preds == 1) & (y == 1)).sum(dim=1).float()
                TN = ((val_preds == 0) & (y == 0)).sum(dim=1).float()
                total_correct = TP + TN
                total_per_sample = y.size(1)  
                sample_accuracies = total_correct / total_per_sample
                correct += torch.sum(sample_accuracies).item()
                val_total += y.size(0) 

        val_acc = correct / val_total
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == epochs-1:
            all_val_preds_array = np.vstack(all_val_preds) 
            classes_with_positive_preds = np.sum(np.sum(all_val_preds_array, axis=0) > 0)
            print(f"- Validation: predicting {classes_with_positive_preds} unique classes out of {num_classes}")

        if use_wandb and wandb.run is not None:
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
    test_total = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            outputs = model(x, lengths)
            targets = y.float().to(device)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_preds = (torch.sigmoid(outputs) > sigmoid).int()
            TP = ((test_preds == 1) & (y == 1)).sum(dim=1).float()
            TN = ((test_preds == 0) & (y == 0)).sum(dim=1).float()
            total_correct = TP + TN
            total_per_sample = y.size(1)
            sample_accuracies = total_correct / total_per_sample
            correct += torch.sum(sample_accuracies).item()
            test_total += y.size(0) 

            all_preds.append(test_preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
    
    test_acc = correct / test_total
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    classes_predicted = np.sum(np.sum(all_preds, axis=0) > 0)
    print(f"- Test: predicting {classes_predicted} unique classes out of {num_classes}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    pos_preds_count = np.sum(all_preds > 0)
    total_preds = all_preds.shape[0] * all_preds.shape[1]
    print(f"- Test: positive predictions: {pos_preds_count}/{total_preds} ({pos_preds_count/total_preds*100:.2f}%)")
    print(f"- Test: classes with positive predictions: {np.sum(np.sum(all_preds, axis=0) > 0)}/{all_preds.shape[1]}")

    sigmoid_outputs = 1/(1+np.exp(-np.array(all_outputs)))
    top_confidence = np.sort(sigmoid_outputs.flatten())[-20:]
    print(f"- Top 20 confidence scores: {top_confidence}")

    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    proba_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy() 
    train_counts = {class_name: np.sum(y_train[:, i]) for i, class_name in enumerate(class_names)}

    detailed_metrics, summary_metrics = evaluate_temporal_model(
        y_proba =proba_outputs,
        y_true = all_labels,
        class_names=class_names,
        output_dir=output_dir,
        model_name=model_name,
        threshold = sigmoid,
        subset=subset,
        short_species=short_species if subset else None,
        medium_species=medium_species if subset else None,
        long_species=long_species if subset else None,
        train_counts=train_counts
)
    
    if use_wandb and wandb.run is not None:
        wandb.log({
            "test/mAP": summary_metrics['mAP'],
            "test/macro_f1": summary_metrics['macro_f1'],
            "model/parameters": num_params
        })

    if use_wandb and wandb.run is not None:
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
    embeddings_dir = "../dataset/dataset_ctx/embeddings_ctx"
    annotations_path = "../dataset/dataset_ctx/embedding_annotations_ctx.csv"
    output_dir = "../results/gru_results_ctx"

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
        epochs = 50
    )
