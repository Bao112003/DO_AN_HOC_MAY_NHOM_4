import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
from datetime import datetime


class CNNClassifier:
    """Class xây dựng và huấn luyện CNN classifier từ đầu"""
    
    def __init__(self, num_classes, img_size=(224, 224)):
        """
        Khởi tạo CNN classifier
        
        Args:
            num_classes: Số lượng classes
            img_size: Kích thước ảnh input
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history_phase1 = None
        self.history_phase2 = None
        
    def build_model(self):
        """
        Xây dựng kiến trúc CNN model
        
        Returns:
            model: Keras model
        """
        self.model = models.Sequential([
            # Block 1
            layers.Conv2D(
                32, (3, 3), 
                activation='relu', 
                input_shape=(*self.img_size, 3),
                padding='same',
                name='conv_1'
            ),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1b'),
            layers.MaxPooling2D((2, 2), name='pool_1'),
            layers.BatchNormalization(name='bn_conv_1'),
            layers.Dropout(0.25, name='dropout_1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2b'),
            layers.MaxPooling2D((2, 2), name='pool_2'),
            layers.BatchNormalization(name='bn_conv_2'),
            layers.Dropout(0.25, name='dropout_2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3b'),
            layers.MaxPooling2D((2, 2), name='pool_3'),
            layers.BatchNormalization(name='bn_conv_3'),
            layers.Dropout(0.25, name='dropout_3'),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4b'),
            layers.MaxPooling2D((2, 2), name='pool_4'),
            layers.BatchNormalization(name='bn_conv_4'),
            layers.Dropout(0.25, name='dropout_4'),
            
            # Fully Connected Layers
            layers.Flatten(name='flatten'),
            
            layers.Dense(
                512, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name='dense_1'
            ),
            layers.BatchNormalization(name='bn_dense_1'),
            layers.Dropout(0.5, name='dropout_dense_1'),
            
            layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name='dense_2'
            ),
            layers.BatchNormalization(name='bn_dense_2'),
            layers.Dropout(0.4, name='dropout_dense_2'),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='Custom_CNN_Model')
        
        return self.model
    
    def print_model_summary(self):
        """In thông tin mô hình"""
        print("\n" + "="*80)
        print(f"{'KIẾN TRÚC MÔ HÌNH CNN':^80}")
        print("="*80)
        self.model.summary()
        
        # Thống kê parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n{'THỐNG KÊ THAM SỐ':^80}")
        print(f"Tổng tham số      : {total_params:,}")
        print(f"Trainable         : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Non-trainable     : {non_trainable_params:,} ({non_trainable_params/total_params*100:.1f}%)")
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model
        
        Args:
            learning_rate: Learning rate cho optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
    
    def create_callbacks(self, save_dir, phase='phase1'):
        """
        Tạo callbacks cho training
        
        Args:
            save_dir: Thư mục lưu model
            phase: 'phase1' hoặc 'phase2'
            
        Returns:
            List callbacks
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if phase == 'phase1':
            patience_early = 7
            patience_lr = 4
            lr_factor = 0.5
            min_lr = 1e-7
            monitor_file = f'{save_dir}/best_cnn_phase1.keras'
        else:
            patience_early = 10
            patience_lr = 5
            lr_factor = 0.3
            min_lr = 1e-8
            monitor_file = f'{save_dir}/best_cnn_final.keras'
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience_early,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=patience_lr,
                min_lr=min_lr,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                monitor_file,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        return callbacks
    
    def train_phase1(self, train_gen, val_gen, epochs=15, 
                     learning_rate=0.001, save_dir='./models',
                     class_weight=None):
        """
        Phase 1: Training với learning rate ban đầu
        
        Args:
            train_gen: Training generator
            val_gen: Validation generator
            epochs: Số epochs
            learning_rate: Learning rate
            save_dir: Thư mục lưu model
            class_weight: Class weights
            
        Returns:
            history: Training history
        """
        print("\n" + "="*80)
        print(f"{'PHASE 1: TRAINING CNN VỚI LEARNING RATE BAN ĐẦU':^80}")
        print("="*80)
        
        # Compile model
        self.compile_model(learning_rate)
        
        # Callbacks
        callbacks = self.create_callbacks(save_dir, phase='phase1')
        
        print(f"\nLearning Rate     : {learning_rate}")
        print(f"Epochs            : {epochs}")
        print(f"Trainable params  : {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Training
        self.history_phase1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Đánh giá
        val_gen.reset()
        results = self.model.evaluate(val_gen, verbose=0)
        
        print(f"\n{'KẾT QUẢ PHASE 1':^80}")
        print(f"Val Loss          : {results[0]:.4f}")
        print(f"Val Accuracy      : {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"Top-3 Accuracy    : {results[2]:.4f} ({results[2]*100:.2f}%)")
        print(f"Precision         : {results[3]:.4f}")
        print(f"Recall            : {results[4]:.4f}")
        
        return self.history_phase1
    
    def train_phase2(self, train_gen, val_gen, epochs=10,
                     learning_rate=0.0001, save_dir='./models',
                     class_weight=None):
        """
        Phase 2: Fine-tuning với learning rate thấp hơn
        
        Args:
            train_gen: Training generator
            val_gen: Validation generator
            epochs: Số epochs
            learning_rate: Learning rate (thấp hơn Phase 1)
            save_dir: Thư mục lưu model
            class_weight: Class weights
            
        Returns:
            history: Training history
        """
        print("\n" + "="*80)
        print(f"{'PHASE 2: FINE-TUNING CNN VỚI LEARNING RATE THẤP':^80}")
        print("="*80)
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        total_params = self.model.count_params()
        
        print(f"Trainable params  : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Compile lại với learning rate thấp
        self.compile_model(learning_rate)
        
        # Callbacks
        callbacks = self.create_callbacks(save_dir, phase='phase2')
        
        print(f"\nLearning Rate     : {learning_rate}")
        print(f"Epochs            : {epochs}")
        
        # Training
        self.history_phase2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Đánh giá
        val_gen.reset()
        results = self.model.evaluate(val_gen, verbose=1)
        
        print(f"\n{'KẾT QUẢ CUỐI CÙNG':^80}")
        print(f"Val Loss          : {results[0]:.4f}")
        print(f"Val Accuracy      : {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"Top-3 Accuracy    : {results[2]:.4f} ({results[2]*100:.2f}%)")
        print(f"Precision         : {results[3]:.4f}")
        print(f"Recall            : {results[4]:.4f}")
        
        f1_score = 2*(results[3]*results[4])/(results[3]+results[4]) if (results[3]+results[4]) > 0 else 0
        print(f"F1-Score          : {f1_score:.4f}")
        
        return self.history_phase2
    
    def save_model(self, filepath):
        """Lưu model"""
        self.model.save(filepath)
        print(f"\nĐã lưu model tại: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"\nĐã load model từ: {filepath}")
        return self.model
    
    def get_combined_history(self):
        """
        Kết hợp history từ 2 phases
        
        Returns:
            combined_history: Dictionary chứa history
        """
        if self.history_phase1 is None:
            return None
        
        combined = {
            'accuracy': self.history_phase1.history['accuracy'],
            'val_accuracy': self.history_phase1.history['val_accuracy'],
            'loss': self.history_phase1.history['loss'],
            'val_loss': self.history_phase1.history['val_loss']
        }
        
        if self.history_phase2 is not None:
            combined['accuracy'] += self.history_phase2.history['accuracy']
            combined['val_accuracy'] += self.history_phase2.history['val_accuracy']
            combined['loss'] += self.history_phase2.history['loss']
            combined['val_loss'] += self.history_phase2.history['val_loss']
        
        return combined
    
    def save_training_history(self, filepath):
        """Lưu training history"""
        history = self.get_combined_history()
        if history is not None:
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Đã lưu training history tại: {filepath}")


if __name__ == "__main__":
    # Demo
    NUM_CLASSES = 10
    
    print("="*80)
    print(f"{'DEMO CNN CLASSIFIER':^80}")
    print("="*80)
    
    classifier = CNNClassifier(num_classes=NUM_CLASSES)
    classifier.build_model()
    classifier.print_model_summary()
    
    print("\n CNN Classifier đã sẵn sàng để training!")
    print("  Sử dụng trong main script với:")
    print("  from model_cnn import CNNClassifier")