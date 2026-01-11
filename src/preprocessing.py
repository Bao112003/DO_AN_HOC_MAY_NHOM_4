import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

class DataPreprocessor:
    """Class xử lý và augmentation dữ liệu"""
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Khởi tạo preprocessor
        
        Args:
            img_size: Kích thước ảnh (height, width)
            batch_size: Số lượng ảnh trong 1 batch
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_gen = None
        self.val_gen = None
        self.class_names = None
        self.num_classes = None
        
    def create_data_generators(self, train_dir, val_dir):
        """
        Tạo data generators cho training và validation
        
        Args:
            train_dir: Đường dẫn thư mục train
            val_dir: Đường dẫn thư mục validation
            
        Returns:
            train_gen, val_gen, class_names, num_classes
        """
        # Data Augmentation cho training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            
            # Geometric Transformations
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            shear_range=0.2,
            horizontal_flip=True,
            
            # Color Augmentation
            brightness_range=[0.7, 1.3],
            channel_shift_range=20.0,
            
            # Fill mode
            fill_mode='nearest',
            data_format='channels_last'
        )
        
        # Validation chỉ preprocessing
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        # Tạo generators
        self.train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Lưu thông tin classes
        self.class_names = list(self.train_gen.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        return self.train_gen, self.val_gen, self.class_names, self.num_classes
    
    def print_dataset_info(self):
        """In thông tin về dataset"""
        print(f"\n{'THÔNG TIN DỮ LIỆU':^80}")
        print(f"Số lượng classes    : {self.num_classes}")
        print(f"Tên các classes     : {self.class_names}")
        print(f"Tổng ảnh train      : {self.train_gen.samples}")
        print(f"Tổng ảnh validation : {self.val_gen.samples}")
        print(f"Steps per epoch     : {self.train_gen.samples // self.batch_size}")
        print(f"Validation steps    : {self.val_gen.samples // self.batch_size}")
        
    def visualize_augmentation(self, num_samples=9):
        """
        Hiển thị mẫu ảnh sau augmentation
        
        Args:
            num_samples: Số lượng ảnh hiển thị
        """
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img, label = next(self.train_gen)
            
            # Denormalize để hiển thị ([-1, 1] → [0, 1])
            img_display = (img[0] + 1) / 2.0
            img_display = np.clip(img_display, 0, 1)
            
            axes[i].imshow(img_display)
            class_idx = np.argmax(label[0])
            axes[i].set_title(
                f'{self.class_names[class_idx]}', 
                fontsize=12, 
                fontweight='bold'
            )
            axes[i].axis('off')
        
        plt.suptitle(
            'Mẫu ảnh sau Data Augmentation', 
            fontsize=16, 
            fontweight='bold'
        )
        plt.tight_layout()
        plt.show()


def get_class_weights(generator):
    """
    Tính class weights để cân bằng dữ liệu
    
    Args:
        generator: Data generator
        
    Returns:
        class_weight_dict: Dictionary chứa weights cho từng class
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    return class_weight_dict, class_weights


def print_class_weights(class_names, class_weights):
    """In class weights"""
    print(f"\n{'CLASS WEIGHTS (Cân bằng dữ liệu)':^80}")
    for class_name, weight in zip(class_names, class_weights):
        print(f"{class_name:20s}: {weight:.4f}")


if __name__ == "__main__":
    # Demo
 
    DATA_DIR = r'E:\DAHM\train'
    VAL_DIR = r'E:\DAHM\val'

    preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
    train_gen, val_gen, class_names, num_classes = preprocessor.create_data_generators(
        DATA_DIR, VAL_DIR
    )
    
    preprocessor.print_dataset_info()
    preprocessor.visualize_augmentation()
    
    class_weight_dict, class_weights = get_class_weights(train_gen)
    print_class_weights(class_names, class_weights)