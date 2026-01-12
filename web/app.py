import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Image Classifier - Compare 3 Models",
    layout="wide"
)

# ==================== LOAD MÔ HÌNH ====================
@st.cache_resource
def load_model(model_path):
    try:
        return keras.models.load_model(model_path)
    except:
        return None

@st.cache_data
def load_class_names(class_names_path):
    import json
    try:
        with open(class_names_path, 'r') as f:
            return json.load(f)
    except:
        return None

# ==================== PREPROCESSING ====================
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ==================== DỰ ĐOÁN ====================
def predict_image(model, img_array, class_names):
    predictions = model.predict(img_array, verbose=0)

    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx]

    all_probs = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    return predicted_class, confidence, all_probs

# ==================== VISUALIZE ====================
def plot_predictions(all_probs, top_k=5, title="Predictions"):
    top_classes = list(all_probs.keys())[:top_k]
    top_probs = [all_probs[c] for c in top_classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_classes, top_probs, color='steelblue')

    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)

    for bar, prob in zip(bars, top_probs):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.2%}", va="center", fontsize=9)

    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================
def main():
    st.title("Image Classification - Compare 3 Models")
    st.markdown("**Test MobileNetV2 vs VGG16 vs CNN**")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")

    # Model paths
    base_dir = st.sidebar.text_input(
        "Models directory",
        value=r"E:\DAHM\models"
    )

    class_names_path = st.sidebar.text_input(
        "Class names path (.json)",
        value=r"E:\DAHM\models\class_names.json"
    )

    # Model selection
    st.sidebar.subheader("Select Models to Test")
    test_mobilenet = st.sidebar.checkbox("MobileNetV2", value=True)
    test_vgg16 = st.sidebar.checkbox("VGG16", value=True)
    test_cnn = st.sidebar.checkbox("CNN", value=True)

    st.sidebar.markdown("---")

    top_k = st.sidebar.slider("Top-K Predictions", 3, 10, 5)

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        0.0, 1.0, 0.5, 0.05
    )

    # Load class names
    class_names = load_class_names(class_names_path)
    if class_names is None:
        st.error(f"Cannot load class names from: {class_names_path}")
        st.stop()
    
    # Load models
    models = {}
    model_paths = {
        "MobileNetV2": os.path.join(base_dir, "mobilenet", "final_model.keras"),
        "VGG16": os.path.join(base_dir, "vgg16", "final_model.keras"),
        "CNN": os.path.join(base_dir, "cnn", "final_model.keras")
    }

    loading_status = st.sidebar.empty()
    loading_status.info("Loading models...")

    if test_mobilenet:
        models["MobileNetV2"] = load_model(model_paths["MobileNetV2"])
    if test_vgg16:
        models["VGG16"] = load_model(model_paths["VGG16"])
    if test_cnn:
        models["CNN"] = load_model(model_paths["CNN"])

    # Check loaded models
    loaded_models = {name: model for name, model in models.items() if model is not None}
    
    if not loaded_models:
        loading_status.error("No models loaded! Check paths.")
        st.stop()
    else:
        loading_status.success(f"Loaded {len(loaded_models)} models: {', '.join(loaded_models.keys())}")

    st.sidebar.markdown("---")
    st.sidebar.info(f"Classes: {len(class_names)}")

    # Main content
    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to classify",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col_info:
            st.write("**Image Information:**")
            st.write(f"- Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- Format: {image.format}")
            st.write(f"- Mode: {image.mode}")

        st.markdown("---")
        st.header("Prediction Results")

        # Preprocess image
        with st.spinner("Preprocessing image..."):
            img_array = preprocess_image(image)

        # Predict with all models
        results = {}
        
        for model_name, model in loaded_models.items():
            with st.spinner(f"Predicting with {model_name}..."):
                predicted_class, confidence, all_probs = predict_image(
                    model, img_array, class_names
                )
                results[model_name] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "probs": all_probs
                }

        # Display results in columns
        if len(loaded_models) == 1:
            cols = st.columns(1)
        elif len(loaded_models) == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)

        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                st.subheader(f"{model_name}")
                
                # Show prediction
                if result["confidence"] >= confidence_threshold:
                    st.success(f"**{result['class']}**")
                else:
                    st.warning(f"**{result['class']}** (Low confidence)")
                
                st.metric("Confidence", f"{result['confidence']:.2%}")

                # Plot top-k predictions
                fig = plot_predictions(
                    result["probs"], 
                    top_k, 
                    title=f"Top-{top_k} - {model_name}"
                )
                st.pyplot(fig)
                plt.close(fig)

                # Show all probabilities
                with st.expander(f"All probabilities - {model_name}"):
                    for cls, prob in result["probs"].items():
                        st.write(f"{cls}: {prob:.4f} ({prob:.2%})")

        # Comparison section
        if len(results) > 1:
            st.markdown("---")
            st.header("Models Comparison")

            # Create comparison table
            import pandas as pd
            
            comparison_data = {
                "Model": [],
                "Prediction": [],
                "Confidence": []
            }

            for model_name, result in results.items():
                comparison_data["Model"].append(model_name)
                comparison_data["Prediction"].append(result["class"])
                comparison_data["Confidence"].append(f"{result['confidence']:.2%}")

            df = pd.DataFrame(comparison_data)
            
            # Highlight best confidence
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]
            
            st.dataframe(df, use_container_width=True)

            # Agreement analysis
            predictions = [result["class"] for result in results.values()]
            if len(set(predictions)) == 1:
                st.success(f"All models agree: **{predictions[0]}**")
            else:
                st.warning(f"All models agree: **{predictions[0]}**")
                # st.warning(f"Models disagree: {', '.join([f'{m}: {r[\"class\"]}' for m, r in results.items()])}")

            # Best confidence
            best_model = max(results.items(), key=lambda x: x[1]["confidence"])
            st.info(f"Highest confidence: **{best_model[0]}** ({best_model[1]['confidence']:.2%})")

    else:
        st.info("Please upload an image to start classification")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Image Classification Demo | MobileNetV2 vs VGG16 vs CNN</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()