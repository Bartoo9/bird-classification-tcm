##### implement a linear baseline nn to check how it performs on the embeddings
## additionally how birdnet performs on the created embeddings, along with a single 128 layer nn

import bioacoustics_model_zoo as bmz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import os
from utils.focal_loss import FocalLoss
from utils.eval_baseline import evaluate_baseline_model
from utils.multi_label_split import multi_label_split
from imblearn.over_sampling import SMOTE
import ast

def birdnet_baseline(embeddings_dir, annotations_path, output_dir=None, 
                     subset=False, scale_weights=False,
                     focal_loss=False, epochs=1000, baseline_model = 'log_reg', use_smote=False):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_annotations = pd.read_csv(annotations_path)

    first_value = all_annotations['species'].iloc[0]
    
    if isinstance(first_value, list):
        print("Species already in list format")
    elif isinstance(first_value, str):
        if first_value.startswith('[') and first_value.endswith(']'):
            print("Converting string representation of lists to actual lists")
            all_annotations['species'] = all_annotations['species'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x]
            )
        else:
            print("Converting plain strings to single-item lists")
            all_annotations['species'] = all_annotations['species'].apply(lambda x: [x] if isinstance(x, str) else x)
    else:
        print(f"Converting unknown type ({type(first_value)}) to lists")
        all_annotations['species'] = all_annotations['species'].apply(lambda x: [str(x)])

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

        selected_species = short_species + medium_species + long_species + ["Noise_Noise"]
        annotations = all_annotations[all_annotations['species'].apply(
            lambda species_list: any(species in selected_species for species in species_list)
        )]
        print(f'using subset of species: {len(selected_species)}')
    else:
        annotations = all_annotations
        print(f'using all species: {len(annotations)}')


    X = []
    y = []
    is_augmented = []
    all_species = set(selected_species) if subset else set()

    for _, row in annotations.iterrows():
        embedding_path = os.path.join(embeddings_dir, row['embedding_name'])
        if os.path.exists(embedding_path):
            X.append(np.load(embedding_path))
            
            species_list = row['species']
            if isinstance(species_list, str):
                species_list = [species_list]
            
            if subset:
                species_list = [species for species in species_list if species in selected_species]
                if len(species_list) == 0:
                    continue
            
            y.append(species_list)
            is_augmented.append('aug_' in row['embedding_name'])
            all_species.update(species_list)  

    X = np.array(X)
    is_augmented = np.array(is_augmented)

    mlb = MultiLabelBinarizer(classes=sorted(all_species))
    y_binary = mlb.fit_transform(y)
    print(f"Number of species in dataset: {len(mlb.classes_)}")

    has_augmented = np.any(is_augmented)

    if has_augmented:
        orig_mask = ~is_augmented
        X_orig = X[orig_mask]
        y_orig = y_binary[orig_mask]

        X_temp_orig, X_test, y_temp_orig, y_test = train_test_split(
            X_orig, y_orig, test_size=0.1, random_state=2)
        
        X_aug = X[is_augmented]
        y_aug = y_binary[is_augmented]

        X_temp = np.vstack([X_temp_orig, X_aug])
        y_temp = np.vstack([y_temp_orig, y_aug])

        if use_smote:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=2, k_neighbors=5)
            X_temp_resampled, y_temp_resampled = smote.fit_resample(X_temp, y_temp)
        else:
            X_temp_resampled, y_temp_resampled = X_temp, y_temp
            
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp_resampled, y_temp_resampled, test_size=0.1, random_state=2)
            
        print("Original class distribution:", np.sum(y_temp, axis=0))
        print("Balanced class distribution:", np.sum(y_temp_resampled, axis=0))
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = multi_label_split(
            X, y_binary, test_size=0.1, val_size=0.1)
        
        X_temp = np.vstack([X_train, X_val])
        y_temp = np.vstack([y_train, y_val])
        X_temp_resampled = X_temp
        y_temp_resampled = y_temp
        
        print("Class distribution (no resampling needed for multi-label):", np.sum(y_temp, axis=0))
        
    print("Original class distribution by class:", np.sum(y_temp, axis=0))
    print("Balanced class distribution by class:", np.sum(y_temp_resampled, axis=0))
    print(f"Original samples: {y_temp.shape[0]}, Balanced samples: {y_temp_resampled.shape[0]}")
    
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if scale_weights:
        pos_counts = np.sum(y_train, axis=0)
        neg_counts = len(y_train) - pos_counts
        class_weights = np.sqrt(neg_counts / (pos_counts + 1e-5)) 
        class_weights = np.clip(class_weights, 0.5, 20.0)  
    else:
        class_weights = None
    
    print("Using balanced class weights")

    birdnet = bmz.BirdNET()

    input_dim = X[0].shape[0]
    num_classes = len(mlb.classes_)

    #models implemeted directly from opensoundscapes
    if baseline_model == 'log_reg':
        hidden_size = None
        birdnet.initialize_custom_classifier(
            classes= mlb.classes_,
            hidden_layer_sizes = [],
        )
        num_params = (input_dim * num_classes) + num_classes
        print('Logistic regression classifier')
    elif baseline_model == 'one_layer':
        hidden_size = 128
        birdnet.initialize_custom_classifier(
            classes= mlb.classes_,
            hidden_layer_sizes = [hidden_size],
        )
        num_params = (input_dim * hidden_size + hidden_size) + (hidden_size * num_classes + num_classes)
        print('One layer classifier')
    elif baseline_model == 'birdnet':
        hidden_size = None
        #REFERENCE: Ben and Dan's implementation
        
        from pathlib import Path
        import pickle
        birdnet_labels_path = "BirdNET_GLOBAL_6K_V2.4_Labels_af.txt"
        BIRDNET_LABELS = Path(birdnet_labels_path).read_text(encoding="utf-8").splitlines()

        birds_list_index = []
        for species_name in mlb.classes_:
            try:
                index = BIRDNET_LABELS.index(species_name)
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
    
    print(f"Classifier architecture: {input_dim} -> {hidden_size if hidden_size else 'direct'} -> {num_classes}")
    print(f"Number of trainable parameters: {num_params}")

    print("Training BirdNet...")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if focal_loss:
        print("Using focal loss")
        criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(class_weights).to(device))

    optimizer = torch.optim.AdamW(birdnet.network.parameters(), 
                                 lr=0.001, weight_decay=0.01)

    #fit it to the data
    from opensoundscape.ml.shallow_classifier import quick_fit

    quick_fit(
        model=birdnet.network,
        train_features=X_train,
        train_labels=y_train,
        validation_features=X_val,
        validation_labels=y_val,
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
        device=device,
        mlb=mlb,
        output_dir=output_dir,
        model_name=f"baseline_{baseline_model}",
        subset=subset,
        threshold=0.5,  
        short_species=[s for s in short_species if s in mlb.classes_] if subset else None,
        medium_species=[s for s in medium_species if s in mlb.classes_] if subset else None,
        long_species=[s for s in long_species if s in mlb.classes_] if subset else None
    )

if __name__ == "__main__":
    embeddings_dir = "../dataset/dataset_ctx_multi/embeddings_multi"
    annotations_path = "../dataset/dataset_ctx_multi/embedding_annotations_multi.csv"
    models = ['log_reg', 'one_layer', 'birdnet']

    for model in models:
        birdnet_baseline(embeddings_dir, annotations_path, subset=True,
                     output_dir="../results/baselines_results/birdnet_baseline_results_ctx",
                     scale_weights=True, focal_loss=True, epochs=1000, baseline_model=model, use_smote=False)

