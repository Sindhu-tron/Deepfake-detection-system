"""
Model evaluation system for deepfake detection
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd

class ModelEvaluator:
    """Comprehensive evaluation for deepfake detection models"""
    
    def __init__(self, model, class_names=['fake', 'real']):
        self.model = model
        self.class_names = class_names
        self.evaluation_results = {}
        
        # Create output directory
        self.output_dir = Path("evaluation_results")
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        
        for directory in [self.output_dir, self.plots_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self, test_dataset, dataset_name="test"):
        """Comprehensive model evaluation"""
        print(f"Evaluating model on {dataset_name} dataset...")
        
        # Get predictions and true labels
        y_true, y_pred, y_pred_proba = self._get_predictions(test_dataset)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Store results
        self.evaluation_results[dataset_name] = {
            'metrics': metrics,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        # Generate reports and plots
        self._create_confusion_matrix(y_true, y_pred, dataset_name)
        self._create_roc_curve(y_true, y_pred_proba, dataset_name)
        self._create_precision_recall_curve(y_true, y_pred_proba, dataset_name)
        self._save_classification_report(y_true, y_pred, dataset_name)
        
        # Print summary
        self._print_evaluation_summary(metrics, dataset_name)
        
        return metrics
    
    def _get_predictions(self, dataset):
        """Get model predictions for dataset"""
        y_true = []
        y_pred_proba = []
        
        print("Generating predictions...")
        
        for batch_images, batch_labels in dataset:
            # Get true labels (convert from categorical to binary)
            true_labels = tf.argmax(batch_labels, axis=1).numpy()
            y_true.extend(true_labels)
            
            # Get predictions
            predictions = self.model.predict(batch_images, verbose=0)
            y_pred_proba.extend(predictions[:, 1])  # Probability of 'real' class
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)  # Binary predictions
        
        return y_true, y_pred, y_pred_proba
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic accuracy
        accuracy = np.mean(y_true == y_pred)
        metrics['accuracy'] = float(accuracy)
        
        # Confusion matrix components
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1_score)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = float(specificity)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            metrics['auc_roc'] = float(auc_roc)
        except:
            metrics['auc_roc'] = 0.0
        
        # AUC-PR
        try:
            auc_pr = average_precision_score(y_true, y_pred_proba)
            metrics['auc_pr'] = float(auc_pr)
        except:
            metrics['auc_pr'] = 0.0
        
        # False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics['false_positive_rate'] = float(fpr)
        metrics['false_negative_rate'] = float(fnr)
        
        # Confusion matrix values
        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
        
        return metrics
    
    def _create_confusion_matrix(self, y_true, y_pred, dataset_name):
        """Create confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'confusion_matrix_{dataset_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roc_curve(self, y_true, y_pred_proba, dataset_name):
        """Create ROC curve visualization"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name.capitalize()} Set')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'roc_curve_{dataset_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not create ROC curve: {e}")
    
    def _create_precision_recall_curve(self, y_true, y_pred_proba, dataset_name):
        """Create Precision-Recall curve visualization"""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR Curve (AP = {avg_precision:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {dataset_name.capitalize()} Set')
            plt.legend(loc="lower left")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'precision_recall_curve_{dataset_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not create PR curve: {e}")
    
    def _save_classification_report(self, y_true, y_pred, dataset_name):
        """Save detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Save as JSON
        with open(self.reports_dir / f'classification_report_{dataset_name}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as readable text
        report_text = classification_report(y_true, y_pred, target_names=self.class_names)
        with open(self.reports_dir / f'classification_report_{dataset_name}.txt', 'w') as f:
            f.write(report_text)
    
    def _print_evaluation_summary(self, metrics, dataset_name):
        """Print evaluation summary"""
        print(f"\n=== {dataset_name.upper()} SET EVALUATION RESULTS ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm['true_negative']}")
        print(f"  False Positives: {cm['false_positive']}")
        print(f"  False Negatives: {cm['false_negative']}")
        print(f"  True Positives: {cm['true_positive']}")
    
    def save_evaluation_results(self):
        """Save all evaluation results to file"""
        with open(self.reports_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved to: {self.output_dir}")
    
    def compare_models(self, model_results):
        """Compare multiple model results"""
        comparison_data = []
        
        for model_name, results in model_results.items():
            for dataset, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Dataset': dataset,
                    'Accuracy': metrics['metrics']['accuracy'],
                    'Precision': metrics['metrics']['precision'],
                    'Recall': metrics['metrics']['recall'],
                    'F1-Score': metrics['metrics']['f1_score'],
                    'AUC-ROC': metrics['metrics']['auc_roc']
                })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(self.reports_dir / 'model_comparison.csv', index=False)
        
        return df

# Test evaluator
if __name__ == "__main__":
    print("Testing model evaluator...")
    
    # This would normally use a trained model
    print("Evaluator class created successfully")