from sklearn.metrics import accuracy_score,auc, roc_curve, f1_score
from sklearn.metrics import average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import pandas as pd 
import seaborn as sns

#temporal model evaluation
def evaluate_temporal_model(y_proba, y_true, class_names, output_dir=None, model_name=None, 
                           threshold=0.4, subset=False, short_species=None, medium_species=None, 
                           long_species=None, train_counts=None):

    warnings.filterwarnings("ignore", message='no positive samples')
    warnings.filterwarnings("ignore", message='no positive class')
    
    if len(y_true) != len(y_proba):
        raise ValueError(f"Length mismatch between y_true ({len(y_true)}) and y_proba ({len(y_proba)})")
    
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        print("Warning: y_proba contains values outside [0,1]. Applying sigmoid.")
        y_proba = 1 / (1 + np.exp(-y_proba))
    
    if isinstance(threshold, (int, float)):
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = np.zeros_like(y_proba, dtype=int)
        for i in range(y_proba.shape[1]):
            y_pred[:, i] = (y_proba[:, i] >= threshold[i]).astype(int)
    
    from sklearn.metrics import hamming_loss, precision_score, recall_score, accuracy_score
    
    samples_f1 = f1_score(y_true, y_pred, average='samples')
    
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    mAP = average_precision_score(y_true, y_proba, average='macro')
    
    from sklearn.metrics import roc_auc_score
    try:
        micro_auroc = roc_auc_score(y_true, y_proba, average='micro')
        macro_auroc = roc_auc_score(y_true, y_proba, average='macro')
    except ValueError:
        micro_auroc = np.nan
        macro_auroc = np.nan
    
    class_metrics = []
    for i, class_name in enumerate(class_names):
        true_binary = y_true[:, i]
        pred_proba = y_proba[:, i]
        pred_binary = y_pred[:, i]
        
        try:
            f1 = f1_score(true_binary, pred_binary, zero_division=0)
            precision = precision_score(true_binary, pred_binary, zero_division=0)
            recall = recall_score(true_binary, pred_binary, zero_division=0)
            ap = average_precision_score(true_binary, pred_proba)
            
            try:
                fpr, tpr, _ = roc_curve(true_binary, pred_proba)
                auroc = auc(fpr, tpr)
            except ValueError:
                auroc = np.nan
        except Exception as e:
            print(f"Error calculating metrics for class {class_name}: {e}")
            f1 = precision = recall = ap = auroc = np.nan
        
        support = np.sum(true_binary)
        
        if subset:
            if class_name in short_species:
                category = 'short'
            elif class_name in medium_species:
                category = 'medium'
            elif class_name in long_species:
                category = 'long'
            else:
                category = 'unknown'
        else:
            category = 'all'
        
        train_support = train_counts.get(class_name, 0) if train_counts else support
        
        class_metrics.append({
            'class': class_name,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'auroc': auroc,
            'support': support,
            'category': category,
            'train_support': train_support
        })
    
    detailed_metrics_df = pd.DataFrame(class_metrics)
    
    if output_dir and model_name:
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_detailed_metrics.csv"), 
                                 index=False)
        
        summary_metrics = {
            'samples_f1': samples_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'mAP': mAP,
            'micro_auroc': micro_auroc,
            'macro_auroc': macro_auroc
        }
        
        pd.DataFrame([summary_metrics]).to_csv(
            os.path.join(output_dir, f"{model_name}_summary_metrics.csv"), 
            index=False
        )
        
        create_multi_temporal_figures(detailed_metrics_df, output_dir, model_name)
        
        create_cooccurrence_matrix(y_true, y_pred, class_names, output_dir, model_name)
        
    return detailed_metrics_df, summary_metrics


def create_cooccurrence_matrix(y_true, y_pred, class_names, output_dir, model_name):
    true_cooccur = np.dot(y_true.T, y_true)
    
    pred_cooccur = np.dot(y_pred.T, y_pred)

    true_diag = np.diag(true_cooccur)
    true_norm = np.zeros_like(true_cooccur, dtype=float) 

    for i in range(len(true_diag)):
        if true_diag[i] > 0:
            true_norm[i, :] = true_cooccur[i, :] / true_diag[i]
    
    pred_diag = np.diag(pred_cooccur)
    pred_norm = np.zeros_like(pred_cooccur, dtype=float)

    for i in range(len(pred_diag)):
        if pred_diag[i] > 0:
            pred_norm[i, :] = pred_cooccur[i, :] / pred_diag[i]
    
    top_cooccur = {}
    for i, class_name in enumerate(class_names):
        if true_diag[i] < 10:  
            continue
        
        cooccur_vals = true_norm[i, :]
        cooccur_classes = np.argsort(cooccur_vals)[::-1][:4] 
        
        top_cooccur[class_name] = [
            (class_names[j], true_norm[i, j]) 
            for j in cooccur_classes if j != i and true_norm[i, j] > 0.1
        ]
    
    significant_threshold = 0.3  
    significant_pairs = []
    
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            if true_diag[i] < 10 or true_diag[j] < 10:
                continue
                
            if true_norm[i, j] > significant_threshold or true_norm[j, i] > significant_threshold:
                significant_pairs.append((i, j, max(true_norm[i, j], true_norm[j, i])))
    
    significant_pairs.sort(key=lambda x: x[2], reverse=True)
    
    top_pairs = significant_pairs[:15]
    
    if top_pairs:
        cooccur_df = pd.DataFrame([
            {'Class 1': class_names[i], 'Class 2': class_names[j], 'Co-occurrence': val}
            for i, j, val in top_pairs
        ])
        
        cooccur_df.to_csv(os.path.join(output_dir, f"{model_name}_cooccurrences.csv"), index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=cooccur_df, x='Co-occurrence', y='Class 1', hue='Class 2', palette='viridis')
        plt.title('Top Species Co-occurrences', fontsize=14)
        plt.xlabel('Co-occurrence Rate', fontsize=12)
        plt.ylabel('Species', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_cooccurrences.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_multi_temporal_figures(metrics_df, output_dir, model_name):

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("pastel")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    colors = {
        'short': '#a6cee3',    
        'medium': '#b2df8a',   
        'long': '#fdbf6f',     
        'unknown': '#95a5a6',  
        'all': '#9b59b6'       
    }
    
    plt.figure(figsize=(10, 6))
    
    categories = metrics_df['category'].unique()
    category_order = {'short': 0, 'medium': 1, 'long': 2}
    categories = sorted(categories, key=lambda x: category_order.get(x, 99))

    for category in categories:
        if category not in colors:
            continue
            
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['ap'],
                   label=f'{category.title()}',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of positive samples', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.title('Average Precision vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_mAP_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    for category in categories:
        if category not in colors:
            continue
            
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['f1'],
                   label=f'{category.title()}',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of positive samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_f1_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    category_means = metrics_df.groupby('category')[['f1', 'ap', 'auroc']].mean()
    
    desired_order = []
    for cat in ['short', 'medium', 'long']:
        if cat in category_means.index:
            desired_order.append(cat)
    
    if 'unknown' in category_means.index:
        desired_order.append('unknown')

    if len(desired_order) > 0:
        category_means = category_means.reindex(desired_order)
        
        light_colors = {
            'f1': '#a8e6cf',     
            'ap': '#dcedc1',      
            'auroc': '#ffd3b6'   
        }

        if all(cat in category_means.index for cat in ['short', 'medium', 'long']):
            category_means = category_means.reindex(['short', 'medium', 'long'])
            
            plt.figure(figsize=(12, 7))
            
            bar_width = 0.25
            r1 = np.arange(len(category_means.index))
            r2 = [x + bar_width for x in r1]
            r3 = [x + bar_width for x in r2]
            
            plt.bar(r1, category_means['f1'], width=bar_width, label='F1 Score', color=light_colors['f1'], edgecolor='black', linewidth=1)
            plt.bar(r2, category_means['ap'], width=bar_width, label='MAP', color=light_colors['ap'], edgecolor='black', linewidth=1)
            plt.bar(r3, category_means['auroc'], width=bar_width, label='AUC', color=light_colors['auroc'], edgecolor='black', linewidth=1)

            plt.xlabel('Bird Call Length Category', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.title('Performance Metrics by Call Length Category', fontsize=16)
            plt.xticks([r + bar_width for r in range(len(category_means.index))], ['Short', 'Medium', 'Long'], fontsize=12)
            plt.ylim(0, 1.0)

            for i, metric in enumerate([category_means['f1'], category_means['ap'], category_means['auroc']]):
                r_pos = [r1, r2, r3][i]
                for j, v in enumerate(metric):
                    plt.text(r_pos[j], v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
            
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{model_name}_category_metrics.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'{model_name}_category_metrics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    summary_stats = metrics_df.groupby('category').agg({
        'f1': ['mean', 'std', 'min', 'max'],
        'ap': ['mean', 'std'],
        'auroc': ['mean', 'std'],
        'support': ['sum', 'mean']
    }).round(3)
    
    summary_stats.columns = [f'{col[0]}_{col[1]}' for col in summary_stats.columns]
    summary_stats.to_csv(os.path.join(output_dir, f"{model_name}_summary_stats.csv"))
    
    return summary_stats