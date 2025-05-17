#stratified sampling ensuring that all segmens from the same recording go to the same split
#this is to prevent leakage between the splits due to the necessary overlap of segments
def stratified_split_by_recording(X_windows, y_encoded, metadata, test_size=0.2, val_size=0.1):

    unique_recordings = metadata['original_file'].unique()
    recording_species = {}

    for recording in unique_recordings:

        recording_mask = metadata['original_file'] == recording

        species_in_recording = set(metadata.loc[recording_mask, 'species'].tolist())
        recording_species[recording] = species_in_recording
    
    species_counts = {}
    for recording, species_set in recording_species.items():
        for species in species_set:
            if species not in species_counts:
                species_counts[species] = 0
            species_counts[species] += 1
    
    recordings_by_rarity = sorted(
        unique_recordings,
        key=lambda r: min([species_counts.get(s, float('inf')) for s in recording_species[r]])
    )
    
    test_recordings = []
    val_recordings = []
    train_recordings = []
    
    current_test_count = {s: 0 for s in species_counts}
    current_val_count = {s: 0 for s in species_counts}
    current_train_count = {s: 0 for s in species_counts}
    
    total = sum(species_counts.values())
    target_test = {s: int(c * test_size) for s, c in species_counts.items()}
    target_val = {s: int(c * val_size) for s, c in species_counts.items()}
    
    for recording in recordings_by_rarity:
        species_set = recording_species[recording]
        
        test_need = sum([max(0, target_test[s] - current_test_count[s]) for s in species_set])
        val_need = sum([max(0, target_val[s] - current_val_count[s]) for s in species_set])
        
        if test_need > 0 and len(test_recordings) < len(unique_recordings) * test_size:
            test_recordings.append(recording)
            for s in species_set:
                current_test_count[s] += 1
        elif val_need > 0 and len(val_recordings) < len(unique_recordings) * val_size:
            val_recordings.append(recording)
            for s in species_set:
                current_val_count[s] += 1
        else:
            train_recordings.append(recording)
            for s in species_set:
                current_train_count[s] += 1
    
    train_mask = metadata['original_file'].isin(train_recordings)
    val_mask = metadata['original_file'].isin(val_recordings)
    test_mask = metadata['original_file'].isin(test_recordings)
    
    train_indices = metadata[train_mask].index.tolist()
    val_indices = metadata[val_mask].index.tolist()
    test_indices = metadata[test_mask].index.tolist()
    
    X_train = [X_windows[i] for i in train_indices]
    X_val = [X_windows[i] for i in val_indices]
    X_test = [X_windows[i] for i in test_indices]
    
    y_train = y_encoded[train_indices]
    y_val = y_encoded[val_indices]
    y_test = y_encoded[test_indices]
    
    print(f"Split by recording: {len(train_recordings)} train, {len(val_recordings)} val, {len(test_recordings)} test")
    print(f"Resulting in {len(X_train)} train, {len(X_val)} val, {len(X_test)} test segments")
    
    return X_train, X_val, X_test, y_train, y_val, y_test