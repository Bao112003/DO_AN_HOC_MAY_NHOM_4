import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st

# ===== MODULE TỰ TẠO =====
from preprocessing import DataPreprocessor, get_class_weights
from eda import EDAAnalyzer
from model_mobilenetv2_PhanNhuBao import MobileNetV2Classifier
from model_cnn_NguyenTuanMinh import CNNClassifier
from model_vgg16_GiaQuy import VGG16Classifier
from evaluation import ModelEvaluator

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Image Classification Training Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Image Classification Training Pipeline")
st.markdown("**So sánh MobileNetV2 vs VGG16 vs CNN**")
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("Configuration")

# Model Selection
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.radio(
    "Chọn mô hình để train:",
    [
        "MobileNetV2 (Transfer Learning)", 
        "VGG16 (Transfer Learning)",
        "CNN (From Scratch)", 
        "All Models (So sánh tất cả)"
    ],
    index=3  # Default: All Models
)

st.sidebar.markdown("---")

# Data Directories
st.sidebar.subheader("Data Directories")
DATA_DIR = st.sidebar.text_input("Train data directory", r"E:\DAHM\train")
VAL_DIR = st.sidebar.text_input("Validation data directory", r"E:\DAHM\val")
SAVE_DIR = st.sidebar.text_input("Model save directory", r"E:\DAHM\models")
RESULTS_DIR = st.sidebar.text_input("Results directory", r"E:\DAHM\results")

st.sidebar.markdown("---")

# Training Hyperparameters
st.sidebar.subheader("Hyperparameters")
IMG_SIZE = st.sidebar.selectbox("Image size", [(224, 224), (160, 160), (128, 128)])
BATCH_SIZE = st.sidebar.slider("Batch size", 8, 64, 32, step=8)

st.sidebar.markdown("**Phase 1 (Initial Training)**")
EPOCHS_PHASE1 = st.sidebar.slider("Phase 1 epochs", 1, 50, 15)
LR1 = st.sidebar.number_input("Phase 1 learning rate", value=0.001, format="%.5f")

st.sidebar.markdown("**Phase 2 (Fine-tuning)**")
EPOCHS_PHASE2 = st.sidebar.slider("Phase 2 epochs", 1, 50, 10)
LR2 = st.sidebar.number_input("Phase 2 learning rate", value=0.0001, format="%.5f")

# Transfer Learning Settings
if "MobileNetV2" in model_type or "VGG16" in model_type or "All Models" in model_type:
    st.sidebar.markdown("**Transfer Learning Settings**")
    FINE_TUNE_AT = st.sidebar.slider("Fine-tune at layer", 0, 150, 100)

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== SESSION STATE ====================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.step = 1  # Auto-start
    st.session_state.mobilenet_trained = False
    st.session_state.vgg16_trained = False
    st.session_state.cnn_trained = False

# ==================== AUTO-RUN PIPELINE ====================
st.header("Training Pipeline Progress")

progress_bar = st.progress(0)
status_text = st.empty()

# ==================== STEP 1: EDA ====================
if st.session_state.step >= 1:
    status_text.text("Step 1/4: Exploratory Data Analysis...")
    progress_bar.progress(10)
    
    st.subheader("Step 1: Exploratory Data Analysis")

    with st.spinner("Running EDA..."):
        eda = EDAAnalyzer(DATA_DIR, VAL_DIR)
        eda.analyze_class_distribution()
        
        # Capture output
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        eda.print_statistics()
        is_imbalanced, ratio = eda.check_imbalance(threshold=2.0)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Display in expander
        with st.expander("Dataset Statistics", expanded=False):
            st.text(output)
        
        eda.plot_class_distribution(
            save_path=f"{RESULTS_DIR}/class_distribution.png"
        )
        eda.plot_train_val_split(
            save_path=f"{RESULTS_DIR}/train_val_split.png"
        )

    st.success("EDA completed")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(f"{RESULTS_DIR}/class_distribution.png"):
            st.image(f"{RESULTS_DIR}/class_distribution.png", caption="Class Distribution")
    with col2:
        if os.path.exists(f"{RESULTS_DIR}/train_val_split.png"):
            st.image(f"{RESULTS_DIR}/train_val_split.png", caption="Train/Val Split")

    st.session_state.step = 2
    progress_bar.progress(20)

# ==================== STEP 2: PREPROCESS ====================
if st.session_state.step >= 2:
    status_text.text("Step 2/4: Data Preprocessing...")
    progress_bar.progress(25)
    
    st.subheader("Step 2: Data Preprocessing")

    with st.spinner("Preparing data generators..."):
        preprocessor = DataPreprocessor(
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        train_gen, val_gen, class_names, num_classes = \
            preprocessor.create_data_generators(DATA_DIR, VAL_DIR)

        class_weight_dict, class_weights = get_class_weights(train_gen)

        st.session_state.train_gen = train_gen
        st.session_state.val_gen = val_gen
        st.session_state.class_names = class_names
        st.session_state.class_weight = class_weight_dict
        st.session_state.num_classes = num_classes

    st.success(f"Loaded {num_classes} classes: {', '.join(class_names)}")
    
    # Display class weights
    with st.expander("Class Weights (for imbalanced data)", expanded=False):
        import pandas as pd
        df_weights = pd.DataFrame({
            'Class': class_names,
            'Weight': class_weights
        })
        st.dataframe(df_weights, use_container_width=True)

    st.session_state.step = 3
    progress_bar.progress(30)

# ==================== STEP 3: MODEL TRAINING ====================
if st.session_state.step >= 3:
    status_text.text("Step 3/4: Model Training...")
    
    st.header("Model Training")
    
    # ========== MobileNetV2 Training ==========
    if model_type in ["MobileNetV2 (Transfer Learning)", "All Models (So sánh tất cả)"]:
        st.subheader("MobileNetV2 (Transfer Learning)")
        
        if not st.session_state.mobilenet_trained:
            progress_bar.progress(35)
            
            # Build Model
            with st.spinner("Building MobileNetV2 model..."):
                mobilenet_classifier = MobileNetV2Classifier(
                    num_classes=st.session_state.num_classes,
                    img_size=IMG_SIZE
                )
                mobilenet_classifier.build_model()
                st.session_state.mobilenet_classifier = mobilenet_classifier
                st.write("Model architecture built")
            
            progress_bar.progress(40)
            
            # Phase 1
            with st.spinner(f"Training Phase 1 (Frozen Base) - {EPOCHS_PHASE1} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE1}, LR: {LR1}")
                
                history1 = st.session_state.mobilenet_classifier.train_phase1(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE1,
                    learning_rate=LR1,
                    save_dir=f"{SAVE_DIR}/mobilenet",
                    class_weight=st.session_state.class_weight
                )
            
            progress_bar.progress(50)
            
            # Phase 2
            with st.spinner(f"Training Phase 2 (Fine-tuning) - {EPOCHS_PHASE2} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE2}, LR: {LR2}")
                
                history2 = st.session_state.mobilenet_classifier.train_phase2(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE2,
                    learning_rate=LR2,
                    fine_tune_at=FINE_TUNE_AT,
                    save_dir=f"{SAVE_DIR}/mobilenet",
                    class_weight=st.session_state.class_weight
                )
            
            # Save
            st.session_state.mobilenet_classifier.save_model(
                f'{SAVE_DIR}/mobilenet/final_model.keras'
            )
            st.session_state.mobilenet_classifier.save_training_history(
                f'{SAVE_DIR}/mobilenet/training_history.json'
            )
            
            st.session_state.mobilenet_trained = True
            st.success("MobileNetV2 training completed")
        else:
            st.info("MobileNetV2 already trained")
    
    # ========== VGG16 Training ==========
    if model_type in ["VGG16 (Transfer Learning)", "All Models (So sánh tất cả)"]:
        st.subheader("VGG16 (Transfer Learning)")
        
        if not st.session_state.vgg16_trained:
            progress_bar.progress(55)
            
            # Build Model
            with st.spinner("Building VGG16 model..."):
                vgg16_classifier = VGG16Classifier(
                    num_classes=st.session_state.num_classes,
                    img_size=IMG_SIZE
                )
                vgg16_classifier.build_model()
                st.session_state.vgg16_classifier = vgg16_classifier
                st.write("Model architecture built")
            
            progress_bar.progress(60)
            
            # Phase 1
            with st.spinner(f"Training Phase 1 (Frozen Base) - {EPOCHS_PHASE1} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE1}, LR: {LR1}")
                
                history1 = st.session_state.vgg16_classifier.train_phase1(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE1,
                    learning_rate=LR1,
                    save_dir=f"{SAVE_DIR}/vgg16",
                    class_weight=st.session_state.class_weight
                )
            
            progress_bar.progress(70)
            
            # Phase 2
            with st.spinner(f"Training Phase 2 (Fine-tuning) - {EPOCHS_PHASE2} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE2}, LR: {LR2}")
                
                history2 = st.session_state.vgg16_classifier.train_phase2(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE2,
                    learning_rate=LR2,
                    fine_tune_at=None,
                    save_dir=f"{SAVE_DIR}/vgg16",
                    class_weight=st.session_state.class_weight
                )
            
            # Save
            st.session_state.vgg16_classifier.save_model(
                f'{SAVE_DIR}/vgg16/final_model.keras'
            )
            st.session_state.vgg16_classifier.save_training_history(
                f'{SAVE_DIR}/vgg16/training_history.json'
            )
            
            st.session_state.vgg16_trained = True
            st.success("VGG16 training completed")
        else:
            st.info("VGG16 already trained")
    
    # ========== CNN Training ==========
    if model_type in ["CNN (From Scratch)", "All Models (So sánh tất cả)"]:
        st.subheader("CNN (From Scratch)")
        
        if not st.session_state.cnn_trained:
            progress_bar.progress(75)
            
            # Build Model
            with st.spinner("Building CNN model..."):
                cnn_classifier = CNNClassifier(
                    num_classes=st.session_state.num_classes,
                    img_size=IMG_SIZE
                )
                cnn_classifier.build_model()
                st.session_state.cnn_classifier = cnn_classifier
                st.write("Model architecture built")
            
            progress_bar.progress(80)
            
            # Phase 1
            with st.spinner(f"Training Phase 1 - {EPOCHS_PHASE1} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE1}, LR: {LR1}")
                
                history1 = st.session_state.cnn_classifier.train_phase1(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE1,
                    learning_rate=LR1,
                    save_dir=f"{SAVE_DIR}/cnn",
                    class_weight=st.session_state.class_weight
                )
            
            progress_bar.progress(85)
            
            # Phase 2
            with st.spinner(f"Training Phase 2 (Lower LR) - {EPOCHS_PHASE2} epochs..."):
                st.write(f"Epochs: {EPOCHS_PHASE2}, LR: {LR2}")
                
                history2 = st.session_state.cnn_classifier.train_phase2(
                    train_gen=st.session_state.train_gen,
                    val_gen=st.session_state.val_gen,
                    epochs=EPOCHS_PHASE2,
                    learning_rate=LR2,
                    save_dir=f"{SAVE_DIR}/cnn",
                    class_weight=st.session_state.class_weight
                )
            
            # Save
            st.session_state.cnn_classifier.save_model(
                f'{SAVE_DIR}/cnn/final_model.keras'
            )
            st.session_state.cnn_classifier.save_training_history(
                f'{SAVE_DIR}/cnn/training_history.json'
            )
            
            st.session_state.cnn_trained = True
            st.success("CNN training completed")
        else:
            st.info("CNN already trained")
    
    st.session_state.step = 4
    progress_bar.progress(90)

# ==================== STEP 4: EVALUATION ====================
if st.session_state.step >= 4:
    status_text.text("Step 4/4: Model Evaluation...")
    
    st.header("Model Evaluation & Comparison")
    
    results_summary = {}
    
    # ========== MobileNetV2 Evaluation ==========
    if st.session_state.mobilenet_trained:
        st.subheader("MobileNetV2 Results")
        
        with st.spinner("Evaluating MobileNetV2..."):
            mobilenet_evaluator = ModelEvaluator(
                model=st.session_state.mobilenet_classifier.model,
                class_names=st.session_state.class_names
            )
            
            mobilenet_history = st.session_state.mobilenet_classifier.get_combined_history()
            
            mobilenet_evaluator.generate_full_report(
                generator=st.session_state.val_gen,
                history=mobilenet_history,
                save_dir=f"{RESULTS_DIR}/mobilenet"
            )
            
            # Get metrics
            st.session_state.val_gen.reset()
            mobilenet_results = st.session_state.mobilenet_classifier.model.evaluate(
                st.session_state.val_gen, 
                verbose=0
            )
            results_summary['MobileNetV2'] = {
                'Loss': mobilenet_results[0],
                'Accuracy': mobilenet_results[1],
                'Top-3 Acc': mobilenet_results[2],
                'Precision': mobilenet_results[3],
                'Recall': mobilenet_results[4],
                'F1-Score': 2*(mobilenet_results[3]*mobilenet_results[4])/(mobilenet_results[3]+mobilenet_results[4])
            }
        
        st.success("MobileNetV2 evaluation completed")
        
        # Display charts
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(f"{RESULTS_DIR}/mobilenet/confusion_matrix.png"):
                st.image(f"{RESULTS_DIR}/mobilenet/confusion_matrix.png", 
                        caption="Confusion Matrix")
        with col2:
            if os.path.exists(f"{RESULTS_DIR}/mobilenet/training_history.png"):
                st.image(f"{RESULTS_DIR}/mobilenet/training_history.png",
                        caption="Training History")
        with col3:
            if os.path.exists(f"{RESULTS_DIR}/mobilenet/f1_scores.png"):
                st.image(f"{RESULTS_DIR}/mobilenet/f1_scores.png",
                        caption="F1-Scores")
    
    # ========== VGG16 Evaluation ==========
    if st.session_state.vgg16_trained:
        st.subheader("VGG16 Results")
        
        with st.spinner("Evaluating VGG16..."):
            vgg16_evaluator = ModelEvaluator(
                model=st.session_state.vgg16_classifier.model,
                class_names=st.session_state.class_names
            )
            
            vgg16_history = st.session_state.vgg16_classifier.get_combined_history()
            
            vgg16_evaluator.generate_full_report(
                generator=st.session_state.val_gen,
                history=vgg16_history,
                save_dir=f"{RESULTS_DIR}/vgg16"
            )
            
            # Get metrics
            st.session_state.val_gen.reset()
            vgg16_results = st.session_state.vgg16_classifier.model.evaluate(
                st.session_state.val_gen, 
                verbose=0
            )
            results_summary['VGG16'] = {
                'Loss': vgg16_results[0],
                'Accuracy': vgg16_results[1],
                'Top-3 Acc': vgg16_results[2],
                'Precision': vgg16_results[3],
                'Recall': vgg16_results[4],
                'F1-Score': 2*(vgg16_results[3]*vgg16_results[4])/(vgg16_results[3]+vgg16_results[4])
            }
        
        st.success("VGG16 evaluation completed")
        
        # Display charts
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(f"{RESULTS_DIR}/vgg16/confusion_matrix.png"):
                st.image(f"{RESULTS_DIR}/vgg16/confusion_matrix.png", 
                        caption="Confusion Matrix")
        with col2:
            if os.path.exists(f"{RESULTS_DIR}/vgg16/training_history.png"):
                st.image(f"{RESULTS_DIR}/vgg16/training_history.png",
                        caption="Training History")
        with col3:
            if os.path.exists(f"{RESULTS_DIR}/vgg16/f1_scores.png"):
                st.image(f"{RESULTS_DIR}/vgg16/f1_scores.png",
                        caption="F1-Scores")
    
    # ========== CNN Evaluation ==========
    if st.session_state.cnn_trained:
        st.subheader("CNN Results")
        
        with st.spinner("Evaluating CNN..."):
            cnn_evaluator = ModelEvaluator(
                model=st.session_state.cnn_classifier.model,
                class_names=st.session_state.class_names
            )
            
            cnn_history = st.session_state.cnn_classifier.get_combined_history()
            
            cnn_evaluator.generate_full_report(
                generator=st.session_state.val_gen,
                history=cnn_history,
                save_dir=f"{RESULTS_DIR}/cnn"
            )
            
            # Get metrics
            st.session_state.val_gen.reset()
            cnn_results = st.session_state.cnn_classifier.model.evaluate(
                st.session_state.val_gen,
                verbose=0
            )
            results_summary['CNN'] = {
                'Loss': cnn_results[0],
                'Accuracy': cnn_results[1],
                'Top-3 Acc': cnn_results[2],
                'Precision': cnn_results[3],
                'Recall': cnn_results[4],
                'F1-Score': 2*(cnn_results[3]*cnn_results[4])/(cnn_results[3]+cnn_results[4])
            }
        
        st.success("CNN evaluation completed")
        
        # Display charts
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(f"{RESULTS_DIR}/cnn/confusion_matrix.png"):
                st.image(f"{RESULTS_DIR}/cnn/confusion_matrix.png",
                        caption="Confusion Matrix")
        with col2:
            if os.path.exists(f"{RESULTS_DIR}/cnn/training_history.png"):
                st.image(f"{RESULTS_DIR}/cnn/training_history.png",
                        caption="Training History")
        with col3:
            if os.path.exists(f"{RESULTS_DIR}/cnn/f1_scores.png"):
                st.image(f"{RESULTS_DIR}/cnn/f1_scores.png",
                        caption="F1-Scores")
    
    progress_bar.progress(95)
    
    # ========== Comparison Table ==========
    if len(results_summary) > 1:
        st.subheader("Model Comparison")
        
        import pandas as pd
        df_comparison = pd.DataFrame(results_summary).T
        
        # Highlight best values
        def highlight_best(s):
            if s.name == 'Loss':
                is_best = s == s.min()
            else:
                is_best = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_best]
        
        styled_df = df_comparison.style.apply(highlight_best, axis=0)
        st.dataframe(styled_df, use_container_width=True)
        
        # Winner
        winner = df_comparison['Accuracy'].idxmax()
        st.success(f"**Best Model: {winner}** (Accuracy: {df_comparison.loc[winner, 'Accuracy']:.4f})")
    
    st.markdown("---")
    
    # Final Summary
    st.subheader("Summary")
    st.info(f"""
    **Training Completed Successfully**
    
    - **Models trained**: {', '.join(results_summary.keys())}
    - **Dataset**: {st.session_state.num_classes} classes, {st.session_state.train_gen.samples} training images
    - **Results saved to**: {RESULTS_DIR}
    - **Models saved to**: {SAVE_DIR}
    
    Check the results folders for detailed metrics, confusion matrices, and training curves.
    """)
    
    progress_bar.progress(100)
    status_text.text("Pipeline completed!")

