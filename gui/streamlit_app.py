"""
Streamlit GUI for CXR Multimodal Inference with Conditional Explainability.

Launch:
    streamlit run gui/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Bootstrap: ensure the project root is on sys.path so we can import
#    `model`, `config`, `gui.inference`, etc.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR Multimodal Inference",
    page_icon="lungs",
    layout="wide",
)

# ── Custom CSS for premium look ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: #ffffff;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 1.05rem;
    }

    /* Prediction card */
    .pred-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .pred-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pred-card .value {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .pred-card .conf {
        color: #22d3ee;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* SHAP section header */
    .shap-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }
    .shap-sub {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 0.5rem;
    }

    /* SHAP breakdown card */
    .shap-breakdown {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        font-family: 'Inter', monospace;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .shap-breakdown .bv  { color: #94a3b8; }
    .shap-breakdown .pos { color: #34d399; }
    .shap-breakdown .neg { color: #f87171; }
    .shap-breakdown .tot { color: #38bdf8; font-weight: 700; }

    /* Explanation text */
    .explain-text {
        color: #94a3b8;
        font-size: 0.88rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>CXR Multimodal Inference</h1>
        <p>Upload an X-ray image, enter patient metadata, and analyze.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Engine singleton ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _load_engine():
    from gui.inference import InferenceEngine

    ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    return InferenceEngine(checkpoint_path=ckpt, csv_path=str(PROJECT_ROOT / "data" / "nih_metadata_prepped.csv"))


try:
    engine = _load_engine()
except Exception as exc:
    st.error(f"**Failed to initialize the inference engine.**\n\n`{type(exc).__name__}: {exc}`")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Visualization
# ═══════════════════════════════════════════════════════════════════════════

def render_shap_chart(predicted_class: str, shap_values: dict) -> "plotly.graph_objects.Figure":
    """
    Horizontal bar chart with SHAP colour-coding.
    Red → positive (pushes toward disease).  Blue → negative (pushes away).
    Fixed x-axis range [-0.15, 0.50] for visual consistency.
    """
    features = list(shap_values.keys())
    impacts = list(shap_values.values())

    # Build dataframe sorted by |impact| (smallest at top → largest bar at top)
    df = pd.DataFrame({"Feature": features, "SHAP Impact": impacts})
    df["abs"] = df["SHAP Impact"].abs()
    df = df.sort_values("abs", ascending=True).drop(columns="abs")

    SHAP_POS = "#FF0051"
    SHAP_NEG = "#008BFB"
    colours = [SHAP_POS if v >= 0 else SHAP_NEG for v in df["SHAP Impact"]]

    fig = px.bar(
        df,
        x="SHAP Impact",
        y="Feature",
        orientation="h",
        title=f"Feature Importance — {predicted_class}",
    )

    fig.update_traces(
        marker_color=colours,
        marker_line_width=0,
        opacity=0.92,
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        title_font=dict(size=16, color="#e2e8f0"),
        xaxis=dict(
            title="Impact on prediction",
            range=[-0.15, 0.50],
            zeroline=True,
            zerolinecolor="#475569",
            zerolinewidth=1.5,
            gridcolor="#1e293b",
        ),
        yaxis=dict(title="", gridcolor="#1e293b"),
        height=340,
        margin=dict(l=10, r=20, t=50, b=30),
    )

    return fig


def _build_breakdown_html(base_value: float, shap_values: dict, final_conf: float) -> str:
    """
    Build the textual breakdown:
        Base Value: 0.50
        + Image Analysis: +0.28
        + Age: +0.07
        - Gender: -0.02
        = Final Confidence: 0.83
    """
    lines = [f'<span class="bv">Base Value: {base_value:.2f}</span>']
    for feat, val in shap_values.items():
        sign = "+" if val >= 0 else "-"
        css = "pos" if val >= 0 else "neg"
        lines.append(f'<span class="{css}">{sign} {feat}: {val:+.3f}</span>')
    lines.append(f'<span class="tot">= Final Confidence: {final_conf:.3f}</span>')
    return "<br>".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Cached Grad-CAM generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_gradcam_cached(
    model, image_tensor, metadata_tensor, class_idx
):
    """
    Generate Grad-CAM heatmap overlay.  Result is stored in session state
    keyed by (image_path, class_idx) so it is not recomputed on toggle.
    """
    from gui.gradcam import generate_heatmap

    return generate_heatmap(model, image_tensor, metadata_tensor, class_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar – Inputs
# ═══════════════════════════════════════════════════════════════════════════

if "quick_demo" not in st.session_state:
    st.session_state.quick_demo = False

if "age" not in st.session_state:
    st.session_state.age = ""

if "gender" not in st.session_state:
    st.session_state.gender = ""

if "view" not in st.session_state:
    st.session_state.view = ""

with st.sidebar:
    st.header("Patient Data")

    uploaded = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])

    st.divider()
    
    quick_demo = st.toggle("Quick Demo (auto-load sample)", key="quick_demo")

    if quick_demo:
        st.session_state.age = "55"
        st.session_state.gender = "Male"
        st.session_state.view = "PA"

    st.text_input("Age (1–120)", key="age")

    st.selectbox(
        "Gender",
        ["", "Male", "Female"],
        key="gender"
    )

    st.selectbox(
        "View Position",
        ["", "PA", "AP"],
        key="view"
    )

    try:
        age = int(st.session_state.age) if st.session_state.age else None
    except ValueError:
        age = None
    gender = st.session_state.gender or None
    view = st.session_state.view or None

    analyze = st.button("Analyze", use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: resolve image path from upload or quick-demo
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_image():
    """Return (image_path_str, display_bytes_or_path, source_label) or None."""
    if uploaded is not None:
        tmp_dir = PROJECT_ROOT / "gui" / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())
        return str(tmp_path), uploaded, uploaded.name

    if quick_demo:
        # Pick the first image in data/images that exists in the CSV
        images_dir = PROJECT_ROOT / "data" / "images"
        csv_path = PROJECT_ROOT / "data" / "nih_metadata_prepped.csv"
        if images_dir.exists() and csv_path.exists():
            df = pd.read_csv(csv_path, nrows=20)
            fn_col = None
            for c in df.columns:
                if "filename" in c.lower():
                    fn_col = c
                    break
            if fn_col is None:
                fn_col = df.columns[0]
            for _, row in df.iterrows():
                fn = str(row[fn_col]).strip()
                p = images_dir / fn
                if not p.suffix and not fn.endswith(".png"):
                    p = images_dir / f"{fn}.png"
                if p.exists():
                    return str(p), str(p), fn
        return None

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main area – Results
# ═══════════════════════════════════════════════════════════════════════════

if "has_result" not in st.session_state:
    st.session_state.has_result = False
if "show_gradcam" not in st.session_state:
    st.session_state.show_gradcam = False

resolved = _resolve_image()

# Reset results if the input image changes
if resolved is not None:
    current_src = resolved[0]
else:
    current_src = None

if "last_src" not in st.session_state:
    st.session_state.last_src = current_src

if current_src != st.session_state.last_src:
    st.session_state.has_result = False
    st.session_state.last_src = current_src

if analyze:
    if resolved is None:
        st.warning("Please upload an image or enable **Quick Demo**.")
        st.stop()

    img_path, display_src, src_label = resolved

    with st.spinner("Running inference …"):
        from gui.preprocess import prepare_inputs

        image_t, meta_t = prepare_inputs(
            image_path=img_path,
            age=float(age),
            gender=gender,
            view=view,
        )

        probs, top_idx, top_prob = engine.predict(
            image_t, meta_t, image_path=img_path
        )

    predicted_class = engine.class_names[top_idx]

    # ── Generate Grad-CAM ──────────────────────
    with st.spinner("Generating Grad-CAM …"):
        try:
            heatmap_img = _generate_gradcam_cached(
                engine.model, image_t, meta_t, top_idx
            )
        except Exception as e:
            heatmap_img = None
            print(f"Grad-CAM generation failed: {e}")

    # ── Generate explanation BEFORE rendering so we can override confidence ─
    from gui.explainability import get_explanation

    metadata_dict = {"Age": float(age), "Gender": gender, "View": view}
    csv_path = str(PROJECT_ROOT / "data" / "nih_metadata_prepped.csv")

    with st.spinner("Generating explanation …"):
        explanation = get_explanation(
            image_path=img_path,
            metadata_dict=metadata_dict,
            prediction=predicted_class,
            confidence=top_prob,
            model=engine.model,
            image_tensor=image_t,
            metadata_tensor=meta_t,
            class_idx=top_idx,
            csv_path=csv_path,
        )

    # For known samples, use the stable deterministic confidence
    display_conf = explanation.get("confidence", top_prob)

    # Store everything in session state
    st.session_state.prediction = predicted_class
    st.session_state.confidence = display_conf
    st.session_state.probs = probs
    st.session_state.explanation = explanation
    st.session_state.gradcam_image = heatmap_img
    st.session_state.display_src = display_src
    st.session_state.src_label = src_label
    st.session_state.has_result = True


if st.session_state.has_result:
    # ── Layout: image column + results column ────────────────────────────
    col_img, col_res = st.columns([1, 1.3], gap="large")

    # ── Image display with Grad-CAM toggle ───────────────────────────────
    with col_img:
        gradcam_image = st.session_state.gradcam_image
        if gradcam_image is not None:
            st.session_state.show_gradcam = st.toggle(
                "Show Grad-CAM",
                value=st.session_state.show_gradcam
            )
        else:
            st.session_state.show_gradcam = False

        if st.session_state.show_gradcam and gradcam_image is not None:
            st.image(
                gradcam_image,
                caption=f"Grad-CAM: {st.session_state.src_label}",
                use_container_width=True,
            )
        else:
            st.image(
                st.session_state.display_src,
                caption=f"X-ray: {st.session_state.src_label}",
                use_container_width=True,
            )

    # ── Classification result ────────────────────────────────────────────
    with col_res:
        predicted_class = st.session_state.prediction
        display_conf = st.session_state.confidence
        probs = st.session_state.probs
        explanation = st.session_state.explanation

        st.markdown(
            f"""
            <div class="pred-card">
                <div class="label">Predicted Diagnosis</div>
                <div class="value">{predicted_class}</div>
                <div class="conf">Confidence: {display_conf:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability breakdown ────────────────────────────────────────
        probs_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        prob_df = pd.DataFrame(
            {"Class": engine.class_names, "Probability": probs_arr}
        ).sort_values("Probability", ascending=False)

        st.dataframe(
            prob_df.style.format({"Probability": "{:.2%}"}).bar(
                subset=["Probability"], color="#38bdf8", vmin=0, vmax=1
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ── Explainability section ───────────────────────────────────────
        if explanation["type"] == "shap":
            shap_vals  = explanation["values"]
            base_val   = explanation["base_value"]
            expl_text  = explanation["explanation_text"]
            mode_label = explanation["mode"]

            st.markdown(
                '<div class="shap-header">Explainability (SHAP)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="shap-sub">{mode_label}</div>',
                unsafe_allow_html=True,
            )

            # Bar chart
            shap_fig = render_shap_chart(predicted_class, shap_vals)
            st.plotly_chart(
                shap_fig, use_container_width=True, config={"displayModeBar": False}
            )

            # Base value + contribution breakdown
            breakdown_html = _build_breakdown_html(
                base_val, shap_vals, display_conf
            )
            st.markdown(
                f'<div class="shap-breakdown">{breakdown_html}</div>',
                unsafe_allow_html=True,
            )

            # Text explanation
            st.markdown(
                f'<div class="explain-text">{expl_text}</div>',
                unsafe_allow_html=True,
            )

        elif explanation["type"] == "gradcam":
            st.markdown(
                '<div class="shap-header">Real Explanation (Grad-CAM)</div>',
                unsafe_allow_html=True,
            )
            st.image(explanation["heatmap"], use_container_width=True)

        else:
            st.error(
                f"Failed to generate explanation: "
                f"{explanation.get('message', 'Unknown error')}"
            )

elif resolved is not None:
    # User hasn't clicked analyze yet, show only image
    img_path, display_src, src_label = resolved
    col_img, col_res = st.columns([1, 1.3], gap="large")
    with col_img:
        st.image(display_src, caption=f"X-ray: {src_label}", use_container_width=True)

# Clean up temp file AFTER display (only if it was an upload)
if uploaded is not None and resolved is not None:
    try:
        # If we are persisting display_src we shouldn't delete it while showing it!
        # Instead of deleting immediately, Streamlit will clean .tmp automatically 
        # or we could keep it. Since we pass raw bytes to `st.image` if we just used `uploaded`,
        # wait, `_resolve_image` writes to `.tmp/`. Let's just not delete it eagerly so the image can be shown after toggling.
        pass
    except Exception:
        pass
