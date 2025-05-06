##### implement a linear baseline nn to check how it performs on the embeddings
## additionallyhow birdnet performs on the embeddings

import bioacoustics_model_zoo as bmz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from utils.focal_loss import FocalLoss
from utils.eval_baseline import evaluate_baseline_model
from imblearn.over_sampling import SMOTE

def birdnet_baseline(embeddings_dir, annotations_path, output_dir=None, 
                     subset=False, scale_weights=False,
                     focal_loss=False, epochs=2000, baseline_model = 'log_reg'):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_annotations = pd.read_csv(annotations_path)

    # selects top 20 species, separated by their vocalization length
    # short species: 1-2s, medium species: 2-3s, long species: 3+ seconds
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
        print(f'using subset of species: {len(selected_species)}')
    else:
        annotations = all_annotations
        print(f'using all species: {len(annotations)}')


    X = []
    y = []
    is_augmented = []

    for _, row in annotations.iterrows():
        embedding_path = os.path.join(embeddings_dir, row['embedding name'])
        if os.path.exists(embedding_path):
            X.append(np.load(embedding_path))
            y.append(row['label'])
            is_augmented.append(row['embedding name'].startswith('aug_'))
    
    X = np.array(X)
    is_augmented = np.array(is_augmented)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    orig_mask = ~is_augmented
    X_orig = X[orig_mask]
    y_orig = y_encoded[orig_mask]

    X_temp_orig, X_test, y_temp_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=0.1, random_state=2, stratify=y_orig)
    
    X_aug = X[is_augmented]
    y_aug = y_encoded[is_augmented]

    X_temp = np.vstack([X_temp_orig, X_aug])
    y_temp = np.concatenate([y_temp_orig, y_aug])

    smote = SMOTE(random_state=2, k_neighbors=5)
    X_temp_resampled, y_temp_resampled = smote.fit_resample(X_temp, y_temp)


    print("Original class distribution:", np.bincount(y_temp))
    print("Balanced class distribution:", np.bincount(y_temp_resampled))

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_resampled, y_temp_resampled, test_size=0.1, random_state=2, stratify=y_temp_resampled)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_resampled, y_temp_resampled, 
        test_size=0.1, random_state=2, stratify=y_temp_resampled)
    
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")

    #robust scaler to normalize the data
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    
    print("Using balanced class weights")

    birdnet = bmz.BirdNET()

    input_dim = X[0].shape[0]
    num_classes = len(np.unique(y_encoded))

    if baseline_model == 'log_reg':
        hidden_size = None
        birdnet.initialize_custom_classifier(
            classes= label_encoder.classes_,
            hidden_layer_sizes = [],
        )
        num_params = (input_dim * num_classes) + num_classes
        print('Logistic regression classifier')
    elif baseline_model == 'one_layer':
        hidden_size = 128
        birdnet.initialize_custom_classifier(
            classes= label_encoder.classes_,
            hidden_layer_sizes = [hidden_size],
        )
        num_params = (input_dim * hidden_size + hidden_size) + (hidden_size * num_classes + num_classes)
        print('One layer classifier')
    elif baseline_model == 'birdnet':
        hidden_size = None
        # referece: Ben and Dan's implementation
        
        from pathlib import Path
        import pickle
        birdnet_labels_path = "BirdNET_GLOBAL_6K_V2.4_Labels_af.txt"
        BIRDNET_LABELS = Path(birdnet_labels_path).read_text(encoding="utf-8").splitlines()
        BIRDNET_LATIN_LABELS = [item.split('_')[0] for item in BIRDNET_LABELS]

        birds_list_index = []
        for species_name in label_encoder.classes_:
            try:
                index = BIRDNET_LATIN_LABELS.index(species_name)
                birds_list_index.append(index)
            except ValueError:
                print(f"Species '{species_name}' not found in BirdNET labels")
        
        birdnet_weights_path = "birdnet_last_layer.pkl"
        
        with open(birdnet_weights_path, "rb") as file:
            loaded_data = pickle.load(file)
        [birdnet_weights, birdnet_bias] = loaded_data

        filtered_weights = birdnet_weights[birds_list_index, :]
        filtered_bias = birdnet_bias[birds_list_index]

        import torch.nn as nn
        import torch

        class PretrainedClassifier(nn.Module):
            def __init__(self, input_size, output_size, weights, bias):
                super(PretrainedClassifier, self).__init__()
                self.linear = nn.Linear(input_size, output_size)
                with torch.no_grad():
                    self.linear.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
                    self.linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
            
            def forward(self, x):
                return self.linear(x)
        
        classifier = PretrainedClassifier(
            input_size=input_dim, 
            output_size=num_classes,
            weights=filtered_weights,
            bias=filtered_bias
        )

        birdnet.use_custom_classifier = True
        birdnet.network = classifier

        num_params = 'Unknown'
        print('Using BirdNet classifier')
    
    else:
        raise ValueError("Invalid baseline model")
    
    #parameters for later comparison
    print(f"Classifier architecture: {input_dim} -> {hidden_size if hidden_size else 'direct'} -> {num_classes}")
    print(f"Number of trainable parameters: {num_params}")

    #some checks
    # print(f"Network type: {type(birdnet.network)}")
    # print(f"Network attributes: {dir(birdnet.network)}")

    #start training the linear classifier or 1 shallow layer birdnet
    print("Training BirdNet...")

    #had to encode the labels for the loss
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_onehot = encoder.transform(y_val.reshape(-1, 1))

    #to device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #depending how aggresive we want scaling to be 
    if scale_weights:
        print("Using scaled class weights")
        class_counts = np.bincount(y_train)
        class_weights = np.sqrt(class_counts.max() / class_counts)

        scale = 0.5
        class_weights = np.power(class_weights, scale)
        class_weights = class_weights/np.mean(class_weights)
        balanced_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Class weights range: {class_weights.min():.2f} to {class_weights.max():.2f}")
    else:
        balanced_weights = None

    if focal_loss:
        print("Using focal loss")
        criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=balanced_weights)

    optimizer = torch.optim.AdamW(birdnet.network.parameters(), 
                                 lr=0.001, weight_decay=0.01)

    #fit it to the data
    from opensoundscape.ml.shallow_classifier import quick_fit

    quick_fit(
        model=birdnet.network,
        train_features=X_train,
        train_labels=y_train_onehot,
        validation_features=X_val,
        validation_labels=y_val_onehot,
        steps=epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    print("Training complete")

    #######################evaluate on the test set
    print("Evaluating on test set...")

    evaluate_baseline_model(
        model=birdnet.network,
        X_test=X_test,
        y_test=y_test,
        encoder=encoder,
        label_encoder=label_encoder,
        device=device,
        output_dir=output_dir,
        model_name=f"baseline_{baseline_model}",
        subset=subset,
        short_species=short_species if subset else None,
        medium_species=medium_species if subset else None,
        long_species=long_species if subset else None
    )

if __name__ == "__main__":
    embeddings_dir = "../dataset/augmented_embeddings"
    annotations_path = "../dataset/_augmented_embedding_annotations.csv"
    models = ['log_reg', 'one_layer', 'birdnet']

    for model in models:
        birdnet_baseline(embeddings_dir, annotations_path, subset=True,
                     output_dir="../results/4_aug_birdnet_baseline_results",
                     scale_weights=True, focal_loss=True, epochs=1000, baseline_model=model)

