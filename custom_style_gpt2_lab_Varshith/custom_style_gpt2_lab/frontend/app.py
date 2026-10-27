"""Streamlit interface for the Custom Style GPT-2 Fine-Tuning Lab."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import (
    CHECKPOINT_DIR,
    load_base_model,
    checkpoint_exists,
    clean_text,
    fine_tune_model,
    generate_text,
    load_fine_tuned_model,
)

SAMPLE_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_dataset.txt"


@st.cache_resource(show_spinner=False)
def get_base_model():
    """Cache the original model for the duration of the Streamlit session."""
    return load_base_model()


def read_sample_dataset() -> str:
    with io.open(SAMPLE_DATASET_PATH, "r", encoding="utf-8") as sample_file:
        return sample_file.read()


st.set_page_config(
    page_title="Custom Style GPT-2 Lab",
    page_icon="🧪",
    layout="wide",
)
st.title("Custom Style GPT-2 Fine-Tuning Lab")
st.write(
    "Fine-tune the real pretrained GPT-2 checkpoint on a small text dataset, "
    "then compare its continuation with the original model."
)

with st.sidebar:
    st.header("Training settings")
    steps = st.slider(
        "Training steps",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Each step is one optimizer update. Start with 5-10 for a classroom demo.",
    )
    learning_rate = st.select_slider(
        "Learning rate",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
        value=5e-5,
        format_func=lambda value: f"{value:g}",
    )
    max_new_tokens = st.slider(
        "Generation length (new tokens)",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
    )

uploaded = st.file_uploader("Upload a custom dataset (.txt)", type=["txt"])
use_sample = st.checkbox("Use the included sample campus-announcement dataset", value=True)

if uploaded is not None:
    raw_text = uploaded.getvalue().decode("utf-8", errors="replace")
    dataset_source = uploaded.name
elif use_sample:
    try:
        raw_text = read_sample_dataset()
        dataset_source = SAMPLE_DATASET_PATH.name
    except OSError as exc:
        raw_text = ""
        dataset_source = ""
        st.error(f"Could not read the sample dataset: {exc}")
else:
    raw_text = ""
    dataset_source = ""

cleaned_text = clean_text(raw_text) if raw_text else ""
if cleaned_text:
    st.subheader("Dataset preview")
    st.caption(f"Source: {dataset_source} | {len(cleaned_text):,} characters after preprocessing")
    st.text_area("Cleaned training text", cleaned_text, height=170)
elif uploaded is not None or not use_sample:
    st.info("Upload a non-empty .txt file or enable the included sample dataset.")

if "training_history" not in st.session_state:
    st.session_state.training_history = []

if st.button("Fine-tune GPT-2", type="primary", disabled=not bool(cleaned_text)):
    progress = st.progress(0, text="Preparing training...")
    status = st.empty()

    def update_progress(completed: int, total: int, loss: float) -> None:
        progress.progress(completed / total, text=f"Step {completed}/{total} | loss {loss:.4f}")

    try:
        status.info("Downloading/loading GPT-2 if needed, then training...")
        result = fine_tune_model(
            cleaned_text,
            steps=steps,
            learning_rate=learning_rate,
            progress_callback=update_progress,
        )
        st.session_state.training_history = result["history"]
        st.session_state.fine_tuned_checkpoint = result["checkpoint"]
        progress.progress(1.0, text="Training complete")
        status.success(f"Checkpoint saved and verified at `{result['checkpoint']}`")
    except Exception as exc:
        progress.empty()
        status.error(f"Training failed: {exc}")
        st.info("Check the internet connection, installed packages, available RAM, and dataset length.")

st.divider()
st.header("Generate and compare")
prompt = st.text_area(
    "Text prompt",
    value="Campus Announcement: The technology club",
    help="Prompts related to the dataset make style differences easier to observe.",
)

if st.button("Generate and compare", disabled=not bool(prompt.strip())):
    if not checkpoint_exists(st.session_state.get("fine_tuned_checkpoint", CHECKPOINT_DIR)):
        st.warning("Fine-tune the model successfully before generating a comparison.")
    else:
        try:
            with st.spinner("Generating from the base and fine-tuned checkpoints..."):
                base_model, base_tokenizer = get_base_model()
                base_output = generate_text(base_model, base_tokenizer, prompt, max_new_tokens)
                fine_model, fine_tokenizer = load_fine_tuned_model(
                    st.session_state.get("fine_tuned_checkpoint", CHECKPOINT_DIR)
                )
                fine_output = generate_text(
                    fine_model, fine_tokenizer, prompt, max_new_tokens
                )
                del fine_model
            left, right = st.columns(2)
            with left:
                st.subheader("Original pretrained GPT-2")
                st.write(base_output)
            with right:
                st.subheader("Fine-tuned GPT-2")
                st.write(fine_output)
            st.markdown(f"**Prompt used:** `{prompt}`")
            if st.session_state.training_history:
                latest_loss = st.session_state.training_history[-1]["loss"]
                st.caption(
                    f"Training run: {len(st.session_state.training_history)} steps | "
                    f"final loss: {latest_loss:.4f}"
                )
            st.caption(
                "Both outputs use the same prompt. Sampling is stochastic, so outputs may "
                "vary between runs."
            )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

history = st.session_state.training_history
if history:
    st.divider()
    st.header("Training diagnostics")
    history_frame = pd.DataFrame(history)
    st.metric("Training steps", len(history))
    st.metric("Final training loss", f"{history[-1]['loss']:.4f}")
    st.line_chart(history_frame.set_index("step")["loss"])
    st.caption(
        "Loss describes how well the model fits the training text; it is not a measure of factuality."
    )
    st.download_button(
        "Download training-loss history (CSV)",
        history_frame.to_csv(index=False),
        "training_loss.csv",
        "text/csv",
    )
    st.info(
        "A small campus dataset may make the fine-tuned model reuse its announcements, "
        "phrasing, and formal event-notice style. Short training runs may show only subtle changes."
    )

with st.expander("Workflow"):
    st.markdown(
        "1. Choose the sample dataset or upload a `.txt` file.\n"
        "2. Select a short training run and click **Fine-tune GPT-2**.\n"
        "3. Confirm the saved checkpoint and inspect the loss chart.\n"
        "4. Enter a prompt and click **Generate and compare**.\n"
        "5. Compare the original and fine-tuned outputs for vocabulary and style."
    )
