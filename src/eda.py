import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

class EDAAnalyzer:
    """Class phân tích EDA cho dataset ảnh"""
    
    def __init__(self, train_dir, val_dir):
        """
        Khởi tạo EDA analyzer
        
        Args:
            train_dir: Đường dẫn thư mục train
            val_dir: Đường dẫn thư mục validation
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.class_names = None
        self.train_distribution = None
        self.val_distribution = None
        
    def analyze_class_distribution(self):
        """Phân tích phân bố số lượng ảnh theo class"""
        # Lấy danh sách classes
        self.class_names = sorted(os.listdir(self.train_dir))
        self.class_names = [c for c in self.class_names if os.path.isdir(
            os.path.join(self.train_dir, c)
        )]
        
        # Đếm số lượng ảnh mỗi class
        train_counts = {}
        val_counts = {}
        
        for class_name in self.class_names:
            # Train
            train_path = os.path.join(self.train_dir, class_name)
            train_counts[class_name] = len([
                f for f in os.listdir(train_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            # Validation
            val_path = os.path.join(self.val_dir, class_name)
            val_counts[class_name] = len([
                f for f in os.listdir(val_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        
        self.train_distribution = train_counts
        self.val_distribution = val_counts
        
        return train_counts, val_counts
    
    def plot_class_distribution(self, save_path=None):
        """
        Vẽ biểu đồ phân bố class
        
        Args:
            save_path: Đường dẫn lưu biểu đồ (optional)
        """
        if self.train_distribution is None:
            self.analyze_class_distribution()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Train distribution
        classes = list(self.train_distribution.keys())
        train_values = list(self.train_distribution.values())
        
        axes[0].bar(classes, train_values, color='steelblue', alpha=0.8)
        axes[0].set_title('Phân bố Training Data', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Số lượng ảnh')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Ghi giá trị lên cột
        for i, v in enumerate(train_values):
            axes[0].text(i, v + max(train_values)*0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        # Validation distribution
        val_values = list(self.val_distribution.values())
        
        axes[1].bar(classes, val_values, color='coral', alpha=0.8)
        axes[1].set_title('Phân bố Validation Data', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Số lượng ảnh')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Ghi giá trị lên cột
        for i, v in enumerate(val_values):
            axes[1].text(i, v + max(val_values)*0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_statistics(self):
        """In thống kê tổng quan"""
        if self.train_distribution is None:
            self.analyze_class_distribution()
        
        train_total = sum(self.train_distribution.values())
        val_total = sum(self.val_distribution.values())
        
        print(f"\n{'THỐNG KÊ DATASET':^80}")
        print("=" * 80)
        print(f"Tổng số classes     : {len(self.class_names)}")
        print(f"Tổng ảnh train      : {train_total}")
        print(f"Tổng ảnh validation : {val_total}")
        print(f"Tổng cộng           : {train_total + val_total}")
        print()
        
        print(f"{'Class':<20} {'Train':>10} {'Val':>10} {'Total':>10} {'Train %':>10}")
        print("-" * 80)
        
        for class_name in self.class_names:
            train_count = self.train_distribution[class_name]
            val_count = self.val_distribution[class_name]
            total = train_count + val_count
            train_pct = (train_count / total) * 100 if total > 0 else 0
            
            print(f"{class_name:<20} {train_count:>10} {val_count:>10} "
                  f"{total:>10} {train_pct:>9.1f}%")
    
    def check_imbalance(self, threshold=2.0):
        """
        Kiểm tra mức độ mất cân bằng dữ liệu
        
        Args:
            threshold: Ngưỡng cảnh báo (ratio max/min)
            
        Returns:
            is_imbalanced: True nếu dữ liệu mất cân bằng
            imbalance_ratio: Tỷ lệ max/min
        """
        if self.train_distribution is None:
            self.analyze_class_distribution()
        
        counts = list(self.train_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        is_imbalanced = imbalance_ratio > threshold
        
        print(f"\n{'PHÂN TÍCH MẤT CÂN BẰNG DỮ LIỆU':^80}")
        print(f"Số ảnh nhiều nhất   : {max_count}")
        print(f"Số ảnh ít nhất      : {min_count}")
        print(f"Tỷ lệ max/min       : {imbalance_ratio:.2f}")
        
        if is_imbalanced:
            print(f"⚠️  CẢNH BÁO: Dữ liệu mất cân bằng (ratio > {threshold})")
            print("   → Nên sử dụng class weights hoặc augmentation")
        else:
            print(f"✓  Dữ liệu tương đối cân bằng (ratio ≤ {threshold})")
        
        return is_imbalanced, imbalance_ratio
    
    def plot_train_val_split(self, save_path=None):
        """
        Vẽ biểu đồ tỷ lệ train/val cho từng class
        
        Args:
            save_path: Đường dẫn lưu biểu đồ (optional)
        """
        if self.train_distribution is None:
            self.analyze_class_distribution()
        
        classes = list(self.train_distribution.keys())
        train_vals = [self.train_distribution[c] for c in classes]
        val_vals = [self.val_distribution[c] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, train_vals, width, label='Train', 
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, val_vals, width, label='Validation', 
                       color='coral', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Số lượng ảnh', fontsize=12)
        ax.set_title('Phân bố Train/Validation theo Class', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Demo
    TRAIN_DIR =  r'E:\DAHM\train'
    VAL_DIR = r'E:\DAHM\val'
  
    eda = EDAAnalyzer(TRAIN_DIR, VAL_DIR)
    eda.analyze_class_distribution()
    eda.print_statistics()
    eda.check_imbalance(threshold=2.0)
    eda.plot_class_distribution()
    eda.plot_train_val_split()