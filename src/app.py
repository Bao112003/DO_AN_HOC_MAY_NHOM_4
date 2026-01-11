import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Image Classifier - MobileNetV2",
    layout="wide"
)

# ==================== LOAD MÔ HÌNH ====================
@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)

@st.cache_data
def load_class_names(class_names_path):
    import json
    with open(class_names_path, 'r') as f:
        return json.load(f)

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
def plot_predictions(all_probs, top_k=5):
    top_classes = list(all_probs.keys())[:top_k]
    top_probs = [all_probs[c] for c in top_classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_classes, top_probs)

    ax.set_xlabel("Confidence")
    ax.set_title(f"Top-{top_k} Predictions")
    ax.set_xlim(0, 1)

    for bar, prob in zip(bars, top_probs):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.2%}", va="center")

    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================
def main():
    st.title("Image Classification with MobileNetV2")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")

    model_path = st.sidebar.text_input(
        "Model path (.keras)",
        value=r"E:\DAHM\models\final_model.keras"
    )

    class_names_path = st.sidebar.text_input(
        "Class names path (.json)",
        value=r"E:\DAHM\models\class_names.json"
    )

    top_k = st.sidebar.slider("Top-K Predictions", 3, 10, 5)

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        0.0, 1.0, 0.5, 0.05
    )

    try:
        model = load_model(model_path)
        class_names = load_class_names(class_names_path)
        st.sidebar.success(f"Loaded model with {len(class_names)} classes")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")

    with col2:
        st.header("Prediction Result")

        if uploaded_file:
            with st.spinner("Processing image..."):
                img_array = preprocess_image(image)
                predicted_class, confidence, all_probs = predict_image(
                    model, img_array, class_names
                )

                if confidence >= confidence_threshold:
                    st.success(f"Prediction: {predicted_class}")
                else:
                    st.warning(f"Prediction: {predicted_class} (low confidence)")

                st.metric("Confidence", f"{confidence:.2%}")

                st.subheader(f"Top-{top_k} Predictions")
                fig = plot_predictions(all_probs, top_k)
                st.pyplot(fig)

                with st.expander("All class probabilities"):
                    for cls, prob in all_probs.items():
                        st.write(f"{cls}: {prob:.4f} ({prob:.2%})")
        else:
            st.info("Please upload an image to start")

if __name__ == "__main__":
    main()
