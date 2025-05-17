import pandas as pd 
import matplotlib.pyplot as plt
import os 
import numpy as np
import seaborn as sns

#evaluation of the multilabel model
#directly used when evaluating after training and validating the model
def evaluate_multilabel_model(y_proba, y_true, class_names, output_dir=None, model_name=None):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss
    
    y_pred = (y_proba > 0.5).astype(int)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    h_loss = hamming_loss(y_true, y_pred)
    
    subset_acc = accuracy_score(y_true, y_pred)
    
    detailed_metrics = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })
    
    detailed_metrics = detailed_metrics.sort_values('support', ascending=False)
    
    summary = {
        'hamming_loss': h_loss,
        'subset_accuracy': subset_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    
    if output_dir and model_name:
        detailed_metrics.to_csv(os.path.join(output_dir, f"{model_name}_multilabel_metrics.csv"), 
                                index=False)
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(class_names)), precision, alpha=0.6, label='Precision')
        plt.bar(range(len(class_names)), recall, alpha=0.6, label='Recall')
        plt.bar(range(len(class_names)), f1, alpha=0.6, label='F1')
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.title('Multi-label Classification Metrics by Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_multilabel_metrics.png"), 
                    dpi=300, bbox_inches='tight')
        
        label_correlations = np.corrcoef(y_true.T)
        plt.figure(figsize=(14, 12))
        sns.heatmap(label_correlations, annot=False, cmap='coolwarm',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Species Co-occurrence Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_label_correlations.png"),
                   dpi=300, bbox_inches='tight')
        
    return detailed_metrics, summary