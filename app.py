import json
import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.config import MODEL_DIR, OUTPUT_DIR
from src.models import build_model, MODEL_NAMES
from src.explainability import image_to_array, predict_batch, make_lime_explanation, make_shap_explanation

st.set_page_config(page_title="ClinXAI", page_icon="👁️", layout="wide")
st.title("👁️ ClinXAI")
st.subheader("Transparent Multi-Model Intraocular Tumor Classification")
st.warning("Educational research prototype only. Not clinically validated; do not use for diagnosis or treatment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"Compute device: {device}")

@st.cache_resource
def load_model(model_name):
    path = MODEL_DIR / f"{model_name}.pt"
    if not path.exists():
        return None, None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(model_name, len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["class_names"]

tab_predict, tab_compare, tab_about = st.tabs(["Prediction & explanations", "Model comparison", "About"])
with tab_predict:
    uploaded = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            arr = image_to_array(image)
            st.image(image, caption="Uploaded image", use_container_width=True)
            available = []
            for name in MODEL_NAMES:
                model, classes = load_model(name)
                if model is None:
                    st.info(f"{name}: model not trained yet.")
                    continue
                probs = predict_batch(np.expand_dims(arr, 0), model, device)[0]
                idx = int(np.argmax(probs))
                available.append((name, model, classes, probs, idx))
                st.metric(name.replace("_", " ").title(), classes[idx],
                          f"Model score: {probs[idx]*100:.1f}%")
            if available:
                selected_name = st.selectbox("Model to explain", [x[0] for x in available])
                chosen = next(x for x in available if x[0] == selected_name)
                _, model, classes, probs, idx = chosen
                st.markdown("#### Class scores (not calibrated clinical probabilities)")
                st.bar_chart(pd.DataFrame({"score": probs}, index=classes))
                method = st.radio("Explanation method", ["LIME", "SHAP"], horizontal=True)
                if st.button("Generate explanation"):
                    with st.spinner("Generating explanation..."):
                        if method == "LIME":
                            overlay, label = make_lime_explanation(arr, model, device)
                            st.image(overlay, caption=f"LIME regions for {classes[label]}", use_container_width=True)
                        else:
                            try:
                                explanation = make_shap_explanation(arr, model, device, classes, max_evals=100)
                                vals = np.asarray(explanation.values)
                                class_idx = int(np.argmax(probs))
                                # SHAP output commonly has shape (1,H,W,C,classes).
                                if vals.ndim == 5:
                                    heat = vals[0, ..., class_idx].mean(axis=-1)
                                elif vals.ndim == 4:
                                    heat = vals[0].mean(axis=-1)
                                else:
                                    st.error(f"Unexpected SHAP output shape: {vals.shape}")
                                    heat = None
                                if heat is not None:
                                    fig, ax = plt.subplots(figsize=(7, 5))
                                    ax.imshow(arr)
                                    ax.imshow(heat, cmap="coolwarm", alpha=0.45)
                                    ax.set_title(f"SHAP contribution view: {classes[class_idx]}")
                                    ax.axis("off")
                                    st.pyplot(fig)
                                    plt.close(fig)
                            except Exception as exc:
                                st.error(f"SHAP explanation failed: {exc}")
                                st.info("SHAP image explanations can be slow and memory-intensive.")
        except Exception as exc:
            st.error(f"Could not process image: {exc}")
    else:
        st.info("Upload a research fundus image. Train models first to enable predictions.")

with tab_compare:
    path = OUTPUT_DIR / "metrics.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        df = pd.DataFrame(metrics).T
        st.markdown("### Held-out test-set metrics")
        st.dataframe(df, use_container_width=True)
        cols = [c for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"] if c in df.columns]
        if cols:
            st.bar_chart(df[cols])
    else:
        st.info("No metrics found. Run `python train.py` after preparing data.")

with tab_about:
    st.markdown("""
    ### Project objectives
    - Compare a custom CNN, ResNet50, and EfficientNet-B0.
    - Explain selected image predictions using LIME and SHAP.
    - Report held-out test metrics and limitations.

    ### Limitations
    - Some dataset classes are benign tumors, not cancers.
    - Image explanations do not prove causation or clinical reasoning.
    - Performance may not generalize to other cameras, hospitals, or populations.
    - This is an educational prototype, not a medical device.
    """)
