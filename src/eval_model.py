import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    accuracy_score, confusion_matrix
)

def evaluate_model(
    model, X_test, y_test, encoder, label_encoder, 
    device, output_dir=None, model_name="model", 
    subset=False, short_species=None, medium_species=None, long_species=None
):
    """
    Comprehensive model evaluation function with visualization and result saving
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model to evaluate
    X_test : numpy.ndarray
        Test set feature embeddings
    y_test : numpy.ndarray
        Test set encoded labels (integers)
    encoder : sklearn.preprocessing.OneHotEncoder
        One-hot encoder for the labels
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder that maps class names to integers
    device : torch.device
        Device to run the model on (CPU or CUDA)
    output_dir : str, optional
        Directory to save results and visualizations
    model_name : str, optional
        Name of the model for saving files
    subset : bool, optional
        Whether a subset of species is being used
    short_species, medium_species, long_species : list, optional
        Lists of species in each length category
    
    Returns:
    --------
    dict
        Dictionary containing all computed metrics
    """
    print(f"Evaluating {model_name} on test set...")

    # Prepare test data
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Get model predictions
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    
    # Convert labels to one-hot encoding
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    # Calculate overall metrics
    map_score = average_precision_score(y_test_onehot, y_pred_probs, average='weighted')
    auroc = roc_auc_score(y_test_onehot, y_pred_probs, average='weighted')
    
    # Calculate per-class metrics
    class_map = average_precision_score(y_test_onehot, y_pred_probs, average=None)
    
    # Generate predictions with threshold
    threshold = 0.5
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # Calculate F1 and accuracy
    f1 = f1_score(y_test_onehot, y_pred_binary, average='weighted')
    accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_pred_binary, axis=1))
    class_f1 = f1_score(y_test_onehot, y_pred_binary, average=None)
    
    # Generate confusion matrix
    y_true_indices = np.argmax(y_test_onehot, axis=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    
    # Print results
    print("\nTest Results:")
    print(f"Mean Average Precision: {map_score:.4f}")
    print(f"AU ROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Collect results in a dictionary
    results = {
        'map': float(map_score),
        'auroc': float(auroc),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'model_name': model_name
    }
    
    # Add per-class metrics
    class_names = label_encoder.classes_
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'map': float(class_map[i]),
            'f1': float(class_f1[i])
        }
    
    results['per_class_metrics'] = per_class_metrics
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save confusion matrix as numpy array
        np.save(os.path.join(output_dir, f'{model_name}_confusion_matrix.npy'), cm)
        
        # Save model weights
        model_path = os.path.join(output_dir, f'{model_name}_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Create and save confusion matrix visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        
        # Plot and save per-class metrics by vocalization length if subset is used
        if subset and short_species and medium_species and long_species:
            # Map species to their lengths
            species_lengths = {}
            for species in short_species:
                species_lengths[species] = 'short'
            for species in medium_species:
                species_lengths[species] = 'medium'
            for species in long_species:
                species_lengths[species] = 'long'
            
            # Create dataframe for plotting
            plot_data = []
            for i, class_name in enumerate(class_names):
                plot_data.append({
                    'Species': class_name,
                    'Length Category': species_lengths.get(class_name, 'unknown'),
                    'MAP Score': class_map[i],
                    'F1 Score': class_f1[i]
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Sort by length category and then by MAP
            length_order = {'short': 0, 'medium': 1, 'long': 2, 'unknown': 3}
            plot_df['Length Order'] = plot_df['Length Category'].map(length_order)
            plot_df = plot_df.sort_values(['Length Order', 'MAP Score'])
            
            # Plot MAP scores
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_df, x='Species', y='MAP Score', 
                           hue='Length Category', palette='viridis')
            plt.xticks(rotation=90)
            plt.title(f'Per-class MAP Scores by Vocalization Length - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_map_by_length.png'))
            
            # Plot F1 scores
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_df, x='Species', y='F1 Score', 
                           hue='Length Category', palette='viridis')
            plt.xticks(rotation=90)
            plt.title(f'Per-class F1 Scores by Vocalization Length - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_f1_by_length.png'))
    
    return results