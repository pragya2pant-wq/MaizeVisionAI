import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="🌽 MaizeVision AI",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    .stApp{
        background: linear-gradient(135deg,#0B1D16,#123524,#1F5135);
        color:white;
    }

    section[data-testid="stSidebar"]{
        background:#10281d;
    }

    h1,h2,h3,h4,h5,h6{
        color:#C8F169;
    }

    .stMetric{
        background:#183A2C;
        padding:18px;
        border-radius:15px;
        border:1px solid #4CAF50;
    }

    div.stButton>button{
        background:#5DBB63;
        color:white;
        border-radius:12px;
        border:none;
    }

    div.stDownloadButton>button{
        background:#6CCF7D;
        color:black;
        border-radius:12px;
        border:none;
    }

    .stDataFrame{
        border-radius:12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model_safe(path):
    return tf.keras.models.load_model(path, compile=False)

models_dict = {
    "custom_cnn": load_model_safe("models/maize_disease_model.keras"),
    "vgg16": load_model_safe("models/best_vgg16.keras")
}

# ---------------- LOAD DATA ----------------

with open("outputs/class_names.json", "r") as f:
    class_names = json.load(f)

history = pd.read_csv(
    "outputs/training_history.csv"
)

with open("outputs/final_metrics.json", "r") as f:
    final_metrics = json.load(f)

# ---------- Prediction History ----------
if "history_predictions" not in st.session_state:
    st.session_state.history_predictions = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌽 MaizeVision AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔍 Disease Detection",
        "📈 Model Performance",
        "📊 Confusion Matrix",
        "🧠 Compare Models",
        "📜 Prediction History",
        "🌽 Dataset",
        "🏗 Model Architecture",
        "⚙ Training Details",
        "🍃 Disease Library",
        "ℹ About"
    ]
)

from pathlib import Path

logo = Path("assets") / "maizelogo.png"

st.sidebar.image(logo, width=120)

st.sidebar.markdown(
    "## 🌽 MaizeVision AI"
)

st.sidebar.caption(
    "Deep Learning for Maize Disease Detection"
)

st.sidebar.markdown("---")

model_map = {
    "Custom CNN": "custom_cnn",
    "VGG16": "vgg16"
}

selected_model_ui = st.sidebar.selectbox(
    "Select Model",
    list(model_map.keys())
)

selected_model = model_map[selected_model_ui]

model = models_dict[selected_model]

st.sidebar.success(
    f"Current Model:\n\n{selected_model_ui}"
)

# ================= HOME =================

if page == "🏠 Home":

    st.markdown(
        """
        <h1 style="text-align:center;color:#C8E66A;">
        🌽 MaizeVision AI
        </h1>

        <h4 style="text-align:center;color:#DCE7D2;">
        Intelligent Deep Learning System for Maize Disease Detection
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.image(
    "assets/maizefield.jpg",
    use_container_width=True
)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌽 Images",
        "4188"
    )

    c2.metric(
        "🌿 Classes",
        "4"
    )

    c3.metric(
        "📈 Validation Accuracy",
        f"{final_metrics['val_accuracy']*100:.2f}%"
    )

    c4.metric(
        "🧠 Model",
        "Custom CNN"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write("""
This application uses Deep Learning to classify maize leaf diseases into four categories.

✔ Blight

✔ Common Rust

✔ Gray Leaf Spot

✔ Healthy

Upload a maize leaf image and receive an instant AI prediction together with confidence scores and model analytics.
""")

    st.subheader("Why Use MaizeVision AI?")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.success("⚡ Fast Prediction")
        st.write("Detect diseases within seconds.")

    with f2:
        st.info("🧠 AI Powered")
        st.write("Built using TensorFlow and Deep Learning.")

    with f3:
        st.warning("🌽 Farmer Friendly")
        st.write("Simple interface requiring only a leaf image.")

    st.subheader("🔄 AI Prediction Workflow")

    workflow = pd.DataFrame({

        "Step": [
            "1",
            "2",
            "3",
            "4",
            "5"
        ],

        "Process": [
            "Upload Leaf Image",
            "Image Preprocessing (128×128)",
            "CNN Feature Extraction",
            "Softmax Classification",
            "Display Prediction & Confidence"
        ]

    })

    st.table(workflow)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("🌿 AI Disease Detection")

        st.write(
            "Upload maize leaf photographs for automatic disease diagnosis."
        )

    with col2:

        st.info("📊 Performance Analytics")

        st.write(
            "Visualize training accuracy, validation accuracy and loss curves."
        )

    with col3:

        st.warning("🧠 Deep Learning")

        st.write(
            "Custom CNN architecture built using TensorFlow and Keras."
        )

    st.markdown("---")

    st.subheader("How the AI Works")

    st.write("""
1. Upload a maize leaf image.

2. The image is resized to **128 × 128** pixels.

3. The CNN extracts visual features such as:
- Spots
- Discoloration
- Texture
- Lesions

4. The neural network predicts the disease.

5. Confidence scores are calculated for all four classes.

6. The disease with the highest confidence is displayed.
""")

    st.markdown("---")

    st.subheader("🏆 Project Highlights")

    highlights = pd.DataFrame({

        "Feature": [
            "Deep Learning",
            "TensorFlow",
            "CNN Architecture",
            "Interactive Dashboard",
            "Real-time Prediction",
            "Model Evaluation"
        ],

        "Status": [
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅"
        ]

    })

    st.table(highlights)

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= DISEASE DETECTION =================

elif page == "🔍 Disease Detection":

    st.title("🌿 AI Disease Detection")

    st.info(f"Using model: **{selected_model}**")

    left, right = st.columns([1, 1])

    with left:

        uploaded_file = st.file_uploader(
            "Upload a maize leaf image",
            type=["jpg", "jpeg", "png"]
        )

        st.caption("Supported formats: JPG, JPEG and PNG")

    with right:

        st.info(
            """
Upload a clear maize leaf image.

The AI model can identify:

• Blight  
• Common Rust  
• Gray Leaf Spot  
• Healthy
"""
        )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )

        # ---------------- PREPROCESS ----------------
        MODEL_INPUT_SIZE = {
            "custom_cnn": (128, 128),
            "vgg16": (224, 224)
        }

        if selected_model not in MODEL_INPUT_SIZE:
            st.error(f"Unknown model selected: {selected_model}")
            st.stop()

        size = MODEL_INPUT_SIZE[selected_model]
        img = image.resize(size)

        img_array = np.array(img.convert("RGB")).astype("float32")

        # safety check
        if np.isnan(img_array).any():
            st.error("Image preprocessing failed (NaN detected)")
            st.stop()

        if selected_model == "vgg16":
            from tensorflow.keras.applications.vgg16 import preprocess_input
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # ---------------- PREDICTION ----------------
        if model is None:
            st.error("Model not loaded properly")
            st.stop()

        prediction = model.predict(img_array, verbose=0)[0]

        prediction = np.array(prediction).reshape(-1)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        # ---------------- SAFETY CHECK ----------------
        if len(prediction) != len(class_names):
            st.error(
                f"Mismatch: model outputs {len(prediction)} classes but dataset has {len(class_names)} classes"
            )
            st.stop()

        # ---------------- HISTORY ----------------
        st.session_state.history_predictions.append({
            "Disease": predicted_class,
            "Confidence (%)": round(confidence * 100, 2),
            "Model": selected_model_ui
        })

        # ---------------- RESULT DISPLAY ----------------
        with col2:

            st.success(f"Prediction: **{predicted_class}**")

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

            if confidence > 0.95:
                st.success("🟢 Very High Confidence")
            elif confidence > 0.80:
                st.info("🟡 Moderate Confidence")
            else:
                st.warning("🔴 Low Confidence")

        st.divider()

        # ---------------- CLASS-WISE CONFIDENCE ----------------
        st.subheader("Class-wise Confidence")

        prob_df = pd.DataFrame({
            "Disease": class_names,
            "Confidence (%)": prediction * 100
        })

        prob_df = prob_df.sort_values(
            by="Confidence (%)",
            ascending=False
        )

        st.bar_chart(prob_df.set_index("Disease"))

        # ---------------- RANKING ----------------
        st.subheader("Confidence Ranking")

        rank_df = prob_df.copy()

        rank_df.insert(
            0,
            "Rank",
            range(1, len(rank_df) + 1)
        )

        st.dataframe(
            rank_df,
            use_container_width=True,
            hide_index=True
        )

        st.dataframe(
            prob_df.style.format({
                "Confidence (%)": "{:.2f}"
            }),
            use_container_width=True
        )

        st.divider()

        # ---------------- SUMMARY ----------------
        st.subheader("Prediction Summary")

        st.info(f"""
### 🌽 Prediction Summary

Disease Detected: **{predicted_class}**

Confidence: **{confidence * 100:.2f}%**

AI Model: **{selected_model}**

The uploaded maize leaf has been classified as **{predicted_class}**.
""")

        report = f"""
MaizeVision AI

Prediction : {predicted_class}

Confidence : {confidence * 100:.2f} %

Model : {selected_model}
"""

        st.download_button(
            "📄 Download Prediction Report",
            report,
            file_name="prediction_report.txt"
        )

        # ---------------- MODEL INFO ----------------
        with st.expander("🧠 Model Information"):

            st.write(f"Selected Model: {selected_model}")
            if selected_model == "custom_cnn":
                st.write("Input Size: 128 × 128")
            else:
                st.write("Input Size: 224 × 224")
            st.write("Framework: TensorFlow / Keras")
            st.write("Output Classes: 4")
            st.write("Activation: Softmax")
            st.write("Optimizer: Adam")

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= MODEL PERFORMANCE =================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Accuracy",
            f"{final_metrics['train_accuracy']*100:.2f}%"
        )

    with col2:
        st.metric(
            "Validation Accuracy",
            f"{final_metrics['val_accuracy']*100:.2f}%"
        )

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Training Loss",
            f"{final_metrics['train_loss']:.4f}"
        )

    with col4:
        st.metric(
            "Validation Loss",
            f"{final_metrics['val_loss']:.4f}"
        )

    st.divider()

    st.subheader("Training Accuracy")
    st.line_chart(history["accuracy"])

    st.subheader("Validation Accuracy")
    st.line_chart(history["val_accuracy"])

    st.subheader("Training Loss")
    st.line_chart(history["loss"])

    st.subheader("Validation Loss")
    st.line_chart(history["val_loss"])

    st.divider()

    st.subheader("Training History")

    st.dataframe(
        history,
        use_container_width=True
    )

    best_epoch = history["val_accuracy"].idxmax() + 1

    best_accuracy = history["val_accuracy"].max()

    st.success(
    f"🏆 Best Validation Accuracy: {best_accuracy*100:.2f}% (Epoch {best_epoch})"
)

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= CONFUSION MATRIX =================

elif page == "📊 Confusion Matrix":

    st.title("📊 Model Evaluation")

    import os

    # ---------- Confusion Matrix ----------

    if os.path.exists("outputs/confusion_matrix.png"):

        st.subheader("Confusion Matrix")

        st.image(
            "outputs/confusion_matrix.png",
            use_container_width=True
        )

    else:
        st.warning("Run evaluate.py to generate the confusion matrix.")

    # ---------- Classification Report ----------

    if os.path.exists("outputs/classification_report.csv"):

        st.subheader("Classification Report")

        report = pd.read_csv(
            "outputs/classification_report.csv",
            index_col=0
        )

        st.dataframe(
            report,
            use_container_width=True
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Classes",
            len(class_names)
        )

        col2.metric(
            "Validation Accuracy",
            f"{final_metrics['val_accuracy']*100:.2f}%"
        )

        col3.metric(
            "Training Accuracy",
            f"{final_metrics['train_accuracy']*100:.2f}%"
        )

    else:

        st.warning("Classification report not found.")

    # ---------- Explanation ----------

    st.divider()

    st.subheader("Understanding the Metrics")

    st.markdown("""

### Precision
Of all images predicted as a disease, how many were correct?

### Recall
Of all actual disease images, how many were correctly detected?

### F1-Score
The harmonic mean of Precision and Recall.

### Support
The number of validation images belonging to each class.

### Confusion Matrix
Shows which diseases were correctly classified and where the model made mistakes.

""")

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= MODEL COMPARISON =================

elif page == "🧠 Compare Models":

    st.title("🧠 AI Model Comparison")

    st.info(
        "Future versions of MaizeVision AI will support multiple deep learning models."
    )

    compare = pd.DataFrame({

        "Model": [
            "Custom CNN",
            "VGG16",
            "MobileNetV2"
        ],

        "Status": [
            "✅ Available",
            "🚧 Coming Soon",
            "🚧 Coming Soon"
        ],

        "Input Size": [
            "128 × 128",
            "224 × 224",
            "224 × 224"
        ],

        "Parameters": [
            "3.3 Million",
            "138 Million",
            "3.5 Million"
        ],

        "Expected Accuracy": [
            "87.3%",
            "90–95%",
            "89–94%"
        ]
    })

    st.dataframe(
        compare,
        use_container_width=True
    )

    st.subheader("Model Comparison")

    st.write("""
    ### Custom CNN
    ✔ Fast

    ✔ Lightweight

    ✔ Trained from scratch

    ---

    ### VGG16

    ✔ Transfer Learning

    ✔ Higher accuracy

    ✔ Larger model

    ---

    ### MobileNetV2

    ✔ Lightweight

    ✔ Mobile deployment

    ✔ Faster inference

    ---

    ### EfficientNetB0

    ✔ Modern architecture

    ✔ Excellent accuracy

    ✔ Efficient computation
    """)

    st.divider()

    st.success("""
Current Version

• Custom CNN is fully implemented.

Future Versions

• VGG16 (Transfer Learning)

• MobileNetV2 (Lightweight Mobile Model)

These models can be trained using the same maize dataset and compared directly inside this dashboard.
""")

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= PREDICTION HISTORY =================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if len(st.session_state.history_predictions) == 0:

        st.info("No predictions have been made yet.")

    else:

        history_df = pd.DataFrame(
            st.session_state.history_predictions
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.download_button(
            "📄 Download Prediction History",
            history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= DATASET =================

elif page == "🌽 Dataset":

    st.title("🌽 Dataset Overview")

    dataset = pd.DataFrame({
        "Disease": [
            "Blight",
            "Common Rust",
            "Gray Leaf Spot",
            "Healthy"
        ],
        "Images": [
            1146,
            1306,
            574,
            1162
        ]
    })

    st.dataframe(
        dataset,
        use_container_width=True
    )

    total_images = 4188

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", total_images)
    col2.metric("Disease Classes", 4)
    col3.metric("Image Size", "128 × 128")

    st.subheader("Dataset Statistics")

    stats = pd.DataFrame({

        "Statistic": [
            "Training Images",
            "Validation Images",
            "Total Images"
        ],

        "Value": [
            3351,
            837,
            4188
        ]

    })

    st.table(stats)

    st.subheader("Dataset Distribution")

    st.bar_chart(
        dataset.set_index("Disease")
    )

    st.subheader("Class Distribution")

    st.pyplot(
        dataset.set_index("Disease")
        .plot.pie(
            y="Images",
            figsize=(6, 6),
            legend=False,
            ylabel=""
        ).figure
    )

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

    st.subheader("Dataset Statistics")

    stats=pd.DataFrame({

    "Statistic":[
    "Training Images",
    "Validation Images",
    "Total Images"
    ],

    "Value":[
    3351,
    837,
    4188
    ]

    })

    st.table(stats)


# ================= MODEL ARCHITECTURE =================

elif page == "🏗 Model Architecture":

    st.title("🏗 CNN Architecture")

    if os.path.exists("outputs/model_summary.txt"):

        with open(
            "outputs/model_summary.txt",
            encoding="utf-8"
        ) as f:

            st.code(
                f.read(),
                language="text"
            )

    else:

        st.warning("model_summary.txt not found.")

    st.markdown("---")

    st.subheader("Model Information")

    info = pd.DataFrame({

        "Property": [
            "Architecture",
            "Framework",
            "Input Size",
            "Output Classes",
            "Optimizer",
            "Loss Function"
        ],

        "Value": [
            "Custom CNN",
            "TensorFlow / Keras",
            "128 × 128",
            "4",
            "Adam",
            "Sparse Categorical Crossentropy"
        ]

    })

    st.table(info)

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= TRAINING DETAILS =================

elif page == "⚙ Training Details":

    st.title("⚙ Training Configuration")

    config = pd.DataFrame({

        "Parameter": [
            "Image Size",
            "Batch Size",
            "Epochs",
            "Optimizer",
            "Loss Function",
            "Activation",
            "Framework"
        ],

        "Value": [
            "128 × 128",
            16,
            10,
            "Adam",
            "Sparse Categorical Crossentropy",
            "Softmax",
            "TensorFlow / Keras"
        ]

    })

    st.table(config)

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= DISEASE LIBRARY =================

elif page == "🍃 Disease Library":

    st.title("🍃 Disease Library")

    st.info(
        "Learn about the four maize leaf classes recognized by MaizeVision AI."
    )

    st.divider()

    with st.expander("🌿 Blight", expanded=True):

        st.image("assets/Corn_Blight (59).jpg")

        st.write("""
**Cause**
- Primarily caused by fungal pathogens.

**Symptoms**
- Long brown or tan lesions.
- Lesions gradually enlarge.
- Leaves dry prematurely.

**Impact**
- Reduces photosynthesis.
- Can significantly reduce crop yield.
""")

    with st.expander("🟠 Common Rust"):

        st.image("assets/Corn_Common_Rust (28).jpg")

        st.write("""
**Cause**
- Caused by *Puccinia sorghi* fungus.

**Symptoms**
- Orange or reddish-brown pustules.
- Appears on both sides of the leaf.
- Easily recognizable during humid weather.

**Impact**
- Moderate yield loss if infection is severe.
""")

    with st.expander("⬜ Gray Leaf Spot"):

        st.image("assets/Corn_Gray_Spot (22).jpg")

        st.write("""
**Cause**
- Caused by *Cercospora zeae-maydis*.

**Symptoms**
- Rectangular gray lesions.
- Lesions follow leaf veins.
- Common in warm and humid environments.

**Impact**
- Severe infections can greatly reduce grain production.
""")

    with st.expander("✅ Healthy Leaf"):

        st.image("assets/Corn_Health (9).jpg")

        st.write("""
**Characteristics**
- Uniform green color.
- No visible lesions.
- No discoloration.
- Normal leaf texture.

Healthy leaves indicate that no disease symptoms were detected by the AI model.
""")

    st.divider()

    st.success(
        "The Custom CNN has been trained to classify all four maize leaf categories."
    )

    st.subheader("Prevention Tips")

    tips = pd.DataFrame({

        "Disease": [
            "Blight",
            "Common Rust",
            "Gray Leaf Spot"
        ],

        "Recommended Action": [
            "Use resistant varieties and fungicides",
            "Monitor fields and remove infected plants",
            "Crop rotation and timely fungicide application"
        ]

    })

    st.table(tips)

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )

# ================= ABOUT =================

elif page == "ℹ About":

    st.title("🌽 About MaizeVision AI")

    st.markdown("## 🎯 Project")

    st.write("""
MaizeVision AI is a Deep Learning application developed for automatic maize leaf disease detection.

The objective is to assist farmers, researchers and students by providing fast and reliable disease identification from leaf images.
""")

    st.divider()

    st.markdown("## 🌽 Dataset")

    st.write("""
**Source:** Kaggle

**Total Images:** 4,188

**Classes**

• Blight

• Common Rust

• Gray Leaf Spot

• Healthy
""")

    st.divider()

    st.markdown("## 🧠 AI Models")

    model_df = pd.DataFrame({

        "Model": [
            "Custom CNN",
            "VGG16",
            "MobileNetV2",
            "EfficientNetB0"
        ],

        "Status": [
            "✅ Implemented",
            "🚧 Coming Soon",
            "🚧 Coming Soon",
            "🚧 Coming Soon"
        ]
    })

    st.table(model_df)

    st.divider()

    st.markdown("## 📊 Final Training Metrics")

    col1, col2 = st.columns(2)

    col1.metric(
        "Training Accuracy",
        f"{final_metrics['train_accuracy']*100:.2f}%"
    )

    col2.metric(
        "Validation Accuracy",
        f"{final_metrics['val_accuracy']*100:.2f}%"
    )

    st.json(final_metrics)

    st.divider()

    st.markdown("## 💻 Technologies Used")

    tech = pd.DataFrame({

        "Technology": [
            "Python",
            "TensorFlow",
            "Keras",
            "NumPy",
            "Pandas",
            "Matplotlib",
            "Streamlit"
        ]
    })

    st.table(tech)

    st.divider()

    st.markdown("## 👩‍💻 Developer")

    st.info("""
This project demonstrates the use of Deep Learning for agricultural disease detection.

It combines image classification, data visualization, model evaluation, and an interactive web interface using Streamlit.
""")

    st.success("""
### 👩‍💻 Developer Information

**Name:** Pragya Pant

**Institute:** IPEC Solutions

**Program:** Artificial Intelligence & Machine Learning

**Career Goal:** Aspiring AI/ML Engineer

**Project:** MaizeVision AI – Maize Leaf Disease Detection using Deep Learning
""")

    st.divider()

    st.subheader("Project Statistics")

    stats = pd.DataFrame({

        "Item": [
            "Framework",
            "Programming Language",
            "Dataset",
            "Disease Classes",
            "Current Model",
            "Deployment"
        ],

        "Value": [
            "TensorFlow / Keras",
            "Python",
            "Kaggle",
            "4",
            "Custom CNN",
            "Streamlit"
        ]

    })

    st.table(stats)

    st.markdown("---")

    st.subheader("💻 System Information")

    system_info = pd.DataFrame({

        "Component": [
            "Operating System",
            "Programming Language",
            "Deep Learning Framework",
            "Web Framework",
            "Image Size",
            "Classes",
            "Deployment"
        ],

        "Details": [
            "Windows",
            "Python 3.x",
            "TensorFlow / Keras",
            "Streamlit",
            "128 × 128",
            "4",
            "Streamlit Community Cloud"
        ]

    })

    st.table(system_info)

    st.success(
        "This application demonstrates an end-to-end Deep Learning pipeline, from image preprocessing and CNN training to model evaluation and web deployment."
    )

    st.markdown("---")

    st.caption(
        "🌽 MaizeVision AI • TensorFlow • Streamlit • 2026"
    )