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
    print(f"Evaluating {model_name} on test set...")

    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    map_score = average_precision_score(y_test_onehot, y_pred_probs, average='weighted')
    auroc = roc_auc_score(y_test_onehot, y_pred_probs, average='weighted')
    
    class_map = average_precision_score(y_test_onehot, y_pred_probs, average=None)
    
    threshold = 0.5
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    f1 = f1_score(y_test_onehot, y_pred_binary, average='weighted')
    accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_pred_binary, axis=1))
    class_f1 = f1_score(y_test_onehot, y_pred_binary, average=None)
    
    y_true_indices = np.argmax(y_test_onehot, axis=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    
    print("\nTest Results:")
    print(f"Mean Average Precision: {map_score:.4f}")
    print(f"AU ROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    results = {
        'map': float(map_score),
        'auroc': float(auroc),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'model_name': model_name
    }
    
    class_names = label_encoder.classes_
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'map': float(class_map[i]),
            'f1': float(class_f1[i])
        }
    
    results['per_class_metrics'] = per_class_metrics
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        #json results
        with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        np.save(os.path.join(output_dir, f'{model_name}_confusion_matrix.npy'), cm)
        

        model_path = os.path.join(output_dir, f'{model_name}_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        #confustion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        
        if subset and short_species and medium_species and long_species:
            species_lengths = {}
            for species in short_species:
                species_lengths[species] = 'short'
            for species in medium_species:
                species_lengths[species] = 'medium'
            for species in long_species:
                species_lengths[species] = 'long'
            
            plot_data = []
            for i, class_name in enumerate(class_names):
                plot_data.append({
                    'Species': class_name,
                    'Length Category': species_lengths.get(class_name, 'unknown'),
                    'MAP Score': class_map[i],
                    'F1 Score': class_f1[i]
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            #length of vocalization
            length_order = {'short': 0, 'medium': 1, 'long': 2, 'unknown': 3}
            plot_df['Length Order'] = plot_df['Length Category'].map(length_order)
            plot_df = plot_df.sort_values(['Length Order', 'MAP Score'])
            
            # map
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_df, x='Species', y='MAP Score', 
                           hue='Length Category', palette='viridis')
            plt.xticks(rotation=90)
            plt.title(f'Per-class MAP Scores by Vocalization Length - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_map_by_length.png'))
            
            #f1
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_df, x='Species', y='F1 Score', 
                           hue='Length Category', palette='viridis')
            plt.xticks(rotation=90)
            plt.title(f'Per-class F1 Scores by Vocalization Length - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_f1_by_length.png'))
    
    return results