
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class ModelEvaluator:
    """Class đánh giá và visualize kết quả mô hình"""
    
    def __init__(self, model, class_names):
        """
        Khởi tạo evaluator
        
        Args:
            model: Keras model
            class_names: Danh sách tên classes
        """
        self.model = model
        self.class_names = class_names
        self.y_true = None
        self.y_pred = None
        self.y_pred_classes = None
        
    def evaluate_on_generator(self, generator):
        """
        Đánh giá model trên generator
        
        Args:
            generator: Data generator
            
        Returns:
            results: Kết quả đánh giá
        """
        generator.reset()
        results = self.model.evaluate(generator, verbose=1)
        
        print(f"\n{'KẾT QUẢ ĐÁNH GIÁ':^80}")
        print("="*80)
        print(f"Loss              : {results[0]:.4f}")
        print(f"Accuracy          : {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"Top-3 Accuracy    : {results[2]:.4f} ({results[2]*100:.2f}%)")
        print(f"Precision         : {results[3]:.4f}")
        print(f"Recall            : {results[4]:.4f}")
        
        f1 = 2*(results[3]*results[4])/(results[3]+results[4]) if (results[3]+results[4]) > 0 else 0
        print(f"F1-Score          : {f1:.4f}")
        
        return results
    
    def predict_on_generator(self, generator):
        """
        Dự đoán trên generator
        
        Args:
            generator: Data generator
        """
        generator.reset()
        self.y_pred = self.model.predict(generator, verbose=1)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)
        self.y_true = generator.classes
        
        return self.y_pred, self.y_pred_classes, self.y_true
    
    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """
        Vẽ confusion matrix
        
        Args:
            save_path: Đường dẫn lưu ảnh
            figsize: Kích thước figure
        """
        if self.y_true is None or self.y_pred_classes is None:
            raise ValueError("Cần gọi predict_on_generator() trước!")
        
        cm = confusion_matrix(self.y_true, self.y_pred_classes)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Số lượng'},
            linewidths=0.5
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_classification_report(self):
        """In classification report chi tiết"""
        if self.y_true is None or self.y_pred_classes is None:
            raise ValueError("Cần gọi predict_on_generator() trước!")
        
        print(f"\n{'CLASSIFICATION REPORT':^80}")
        print("="*80)
        
        report = classification_report(
            self.y_true, 
            self.y_pred_classes,
            target_names=self.class_names,
            digits=4
        )
        
        print(report)
        
        return report
    
    def plot_f1_scores_per_class(self, save_path=None):
        """
        Vẽ biểu đồ F1-score theo từng class
        
        Args:
            save_path: Đường dẫn lưu ảnh
        """
        if self.y_true is None or self.y_pred_classes is None:
            raise ValueError("Cần gọi predict_on_generator() trước!")
        
        # Tạo classification report dạng dict
        report_dict = classification_report(
            self.y_true,
            self.y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Chuyển sang DataFrame
        df_report = pd.DataFrame(report_dict).transpose()
        
        # Lấy F1-score của từng class
        f1_scores = df_report.loc[self.class_names, 'f1-score']
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, f1_scores, color='steelblue', alpha=0.8)
        
        plt.title('F1-Score theo từng Class', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Ghi giá trị lên cột
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval + 0.01,
                f'{yval:.2f}',
                ha='center', 
                va='bottom', 
                fontsize=10,
                fontweight='bold'
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Vẽ biểu đồ training history
        
        Args:
            history: Dictionary chứa training history
            save_path: Đường dẫn lưu ảnh
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        epochs_range = range(len(history['accuracy']))
        
        # ================= Accuracy =================
        axes[0, 0].plot(
            history['accuracy'], 
            label='Train', 
            linewidth=2, 
            marker='o', 
            markersize=4
        )
        axes[0, 0].plot(
            history['val_accuracy'], 
            label='Validation', 
            linewidth=2, 
            marker='s', 
            markersize=4
        )
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ================= Loss =================
        axes[0, 1].plot(
            history['loss'], 
            label='Train', 
            linewidth=2, 
            marker='o', 
            markersize=4
        )
        axes[0, 1].plot(
            history['val_loss'], 
            label='Validation', 
            linewidth=2, 
            marker='s', 
            markersize=4
        )
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ================= Overfitting Gap =================
        train_val_gap = np.array(history['accuracy']) - np.array(history['val_accuracy'])
        
        axes[1, 0].plot(
            epochs_range, 
            train_val_gap, 
            linewidth=2, 
            marker='o', 
            markersize=4, 
            color='red', 
            label='Gap'
        )
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.05, color='orange', linestyle=':', alpha=0.5, label='5% threshold')
        axes[1, 0].axhline(y=0.10, color='red', linestyle=':', alpha=0.5, label='10% threshold')
        
        axes[1, 0].set_title('Overfitting Gap (Train - Val Accuracy)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Difference')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Annotation
        final_gap = train_val_gap[-1]
        gap_color = 'green' if final_gap < 0.05 else ('orange' if final_gap < 0.10 else 'red')
        gap_status = 'Tốt' if final_gap < 0.05 else ('Chấp nhận được' if final_gap < 0.10 else 'Overfitting')
        
        axes[1, 0].text(
            len(train_val_gap) - 1, 
            final_gap,
            f'{final_gap:.3f}\n{gap_status}',
            fontsize=10,
            color=gap_color,
            fontweight='bold',
            ha='right',
            va='bottom'
        )
        
        # ================= Statistics =================
        axes[1, 1].axis('off')
        
        stats_text = f"""
        THỐNG KÊ TRAINING
        
        Best Train Accuracy    : {max(history['accuracy']):.4f}
        Best Val Accuracy      : {max(history['val_accuracy']):.4f}
        
        Final Train Accuracy   : {history['accuracy'][-1]:.4f}
        Final Val Accuracy     : {history['val_accuracy'][-1]:.4f}
        
        Min Train Loss         : {min(history['loss']):.4f}
        Min Val Loss           : {min(history['val_loss']):.4f}
        
        Final Train Loss       : {history['loss'][-1]:.4f}
        Final Val Loss         : {history['val_loss'][-1]:.4f}
        
        Overfitting Gap        : {final_gap:.4f}
        Status                 : {gap_status}
        """
        
        axes[1, 1].text(
            0.1, 0.9, 
            stats_text,
            fontsize=11,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # In thống kê
        print("\n" + "="*80)
        print(f"{'PHÂN TÍCH TRAINING CURVES':^80}")
        print("="*80)
        print(f"Best Train Accuracy   : {max(history['accuracy']):.4f}")
        print(f"Best Val Accuracy     : {max(history['val_accuracy']):.4f}")
        print(f"Final Train Accuracy  : {history['accuracy'][-1]:.4f}")
        print(f"Final Val Accuracy    : {history['val_accuracy'][-1]:.4f}")
        print(f"Final Overfitting Gap : {final_gap:.4f} ({gap_status})")
        print(f"Min Val Loss          : {min(history['val_loss']):.4f}")
        print(f"Final Val Loss        : {history['val_loss'][-1]:.4f}")
    
    def generate_full_report(self, generator, history=None, save_dir='./results'):
        """
        Tạo báo cáo đánh giá đầy đủ
        
        Args:
            generator: Data generator
            history: Training history (optional)
            save_dir: Thư mục lưu kết quả
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"{'TẠO BÁO CÁO ĐÁNH GIÁ':^80}")
        print("="*80)
        
        # 1. Evaluate
        print("\n1. Đánh giá model...")
        self.evaluate_on_generator(generator)
        
        # 2. Predict
        print("\n2. Dự đoán trên validation set...")
        self.predict_on_generator(generator)
        
        # 3. Confusion Matrix
        print("\n3. Vẽ Confusion Matrix...")
        self.plot_confusion_matrix(save_path=f'{save_dir}/confusion_matrix.png')
        
        # 4. Classification Report
        print("\n4. In Classification Report...")
        self.print_classification_report()
        
        # 5. F1-scores
        print("\n5. Vẽ F1-scores theo class...")
        self.plot_f1_scores_per_class(save_path=f'{save_dir}/f1_scores.png')
        
        # 6. Training History
        if history is not None:
            print("\n6. Vẽ Training History...")
            self.plot_training_history(history, save_path=f'{save_dir}/training_history.png')
        
        print("\n" + "="*80)
        print(f"{'HOÀN THÀNH BÁO CÁO':^80}")
        print(f"Kết quả đã được lưu tại: {save_dir}")
        print("="*80)


if __name__ == "__main__":
    # Demo
    print("Module evaluation.py - Sử dụng trong main script")