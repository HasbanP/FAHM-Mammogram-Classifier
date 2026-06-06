"""
FAHM Mammogram Classifier
Streamlit + EfficientNet-B0 + GradCAM

Deployment Ready for Hugging Face Spaces
"""

import streamlit as st
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="FAHM Mammogram Classifier",
    page_icon="🩺",
    layout="wide"
)


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()


# ------------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_image(image_pil):

    input_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)[0]

    malignant_prob = round(float(probs[1]) * 100, 2)

    benign_prob = round(float(probs[0]) * 100, 2)

    threshold = 0.40

    classification = (
        "Malignant"
        if float(probs[1]) >= threshold
        else "Benign/Normal"
    )

    return {
        "classification": classification,
        "confidence": f"{max(malignant_prob, benign_prob):.1f}%",
        "malignant_probability": f"{malignant_prob:.1f}%",
        "benign_probability": f"{benign_prob:.1f}%",
        "threshold_used": threshold
    }


# ------------------------------------------------------------
# GRADCAM
# ------------------------------------------------------------

def generate_gradcam(image_pil):

    raw_image = image_pil.resize((224, 224))

    raw_np = np.array(raw_image).astype(np.float32) / 255.0

    input_tensor = transform(raw_image).unsqueeze(0)

    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(1)]
    )[0]

    visualization = show_cam_on_image(
        raw_np,
        grayscale_cam,
        use_rgb=True
    )

    return raw_image, visualization, grayscale_cam


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🩺 FAHM Mammogram Classifier")

st.subheader(
    "AI-Assisted Early Breast Cancer Screening"
)

st.markdown(
    """
> ⚠️ **Clinical Disclaimer**

This tool is intended solely as a research and educational screening aid.

All predictions must be reviewed by a qualified radiologist.

This application is not a medical device and must not be used as a standalone diagnostic system.
"""
)

st.divider()


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Mammogram Image",
    type=["jpg", "jpeg", "png", "pgm"],
    help="Supported formats: JPG, JPEG, PNG, PGM"
)


# ------------------------------------------------------------
# MAIN WORKFLOW
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        st.divider()

        # ----------------------------------------------------
        # ROW 1
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Uploaded Mammogram")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Filename: {uploaded_file.name}"
            )

        with col2:

            st.subheader("🔬 Analysis Result")

            with st.spinner(
                "Analysing mammogram..."
            ):

                result = predict_image(image)

                classification = result[
                    "classification"
                ]

                if classification == "Malignant":

                    st.error(
                        "⚠️ Malignant",
                        icon="🔴"
                    )

                    st.markdown(
                        "**Immediate radiologist review recommended.**"
                    )

                else:

                    st.success(
                        "✅ Benign/Normal",
                        icon="🟢"
                    )

                    st.markdown(
                        "**No malignant features detected. Routine follow-up recommended.**"
                    )

                st.metric(
                    "Confidence Score",
                    result["confidence"]
                )

                mal_prob = float(
                    result["malignant_probability"]
                    .replace("%", "")
                )

                ben_prob = float(
                    result["benign_probability"]
                    .replace("%", "")
                )

                st.markdown(
                    "**Probability Breakdown**"
                )

                st.progress(
                    int(mal_prob),
                    text=f"🔴 Malignant: {result['malignant_probability']}"
                )

                st.progress(
                    int(ben_prob),
                    text=f"🟢 Benign/Normal: {result['benign_probability']}"
                )

                st.caption(
                    f"Classification threshold: {result['threshold_used']} "
                    f"(optimised for screening sensitivity)"
                )

        st.divider()

        # ----------------------------------------------------
        # ROW 2 - GRADCAM
        # ----------------------------------------------------

        st.subheader(
            "🔍 GradCAM Visualisation"
        )

        st.markdown(
            """
The heatmap highlights regions where the model focused while generating its prediction.

- Red/Yellow = higher attention
- Blue = lower attention
"""
        )

        with st.spinner(
            "Generating GradCAM..."
        ):

            raw_img, gradcam_img, grayscale = (
                generate_gradcam(image)
            )

            col3, col4, col5 = st.columns(3)

            with col3:

                st.markdown(
                    "**Original Image (224×224)**"
                )

                st.image(
                    raw_img,
                    use_container_width=True
                )

            with col4:

                st.markdown(
                    "**GradCAM Overlay**"
                )

                st.image(
                    gradcam_img,
                    use_container_width=True
                )

            with col5:

                st.markdown(
                    "**Attention Intensity Map**"
                )

                fig, ax = plt.subplots(
                    figsize=(4, 4)
                )

                im = ax.imshow(
                    grayscale,
                    cmap="jet",
                    vmin=0,
                    vmax=1
                )

                plt.colorbar(
                    im,
                    ax=ax,
                    label="Attention Intensity"
                )

                ax.axis("off")

                st.pyplot(fig)

                plt.close()

        st.caption(
            """
⚠️ GradCAM is an interpretability tool only.

Highlighted regions indicate model attention and do not constitute a clinical diagnosis.

Model localisation reliability is limited by dataset size and should not be interpreted as a confirmed lesion location.
"""
        )

        st.divider()

        # ----------------------------------------------------
        # PRIVACY
        # ----------------------------------------------------

        st.subheader(
            "🔒 Privacy & Compliance"
        )

        col6, col7 = st.columns(2)

        with col6:

            st.info(
                """
**Data Security**

- Image processed in memory only
- No image storage
- No patient data collection
- Stateless processing
"""
            )

        with col7:

            st.info(
                """
**Privacy Compliance**

- No identifiers collected
- No logging of uploaded images
- No post-inference retention
- Research-focused architecture
"""
            )

    except Exception as e:

        st.error(
            f"Processing failed:\n\n{str(e)}"
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.caption("🏥 FAHM Biotechnology")

with c2:
    st.caption(
        "🤖 EfficientNet-B0 + GradCAM"
    )

with c3:
    st.caption(
        "🎓 Research & Educational Use"
    )