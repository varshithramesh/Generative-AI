# Custom Style GPT-2 Fine-Tuning Lab

A local Streamlit demonstration that fine-tunes the real Hugging Face `gpt2`
checkpoint on a small campus-announcement dataset using PyTorch, then compares
the original and fine-tuned models.

The local virtual environments and generated model checkpoint are intentionally
excluded from source packages because they are large generated/dependency files.
Create a fresh `.venv` and run the setup steps below when preparing a new copy.

## Project layout

```text
backend\             Model loading, training, tokenization, and generation
frontend\app.py      Streamlit user interface
data\                Sample training datasets
outputs\             Generated fine-tuned checkpoints
requirements.txt     Python dependencies
```

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 (64-bit recommended)
- Internet access for the first download of GPT-2 from Hugging Face
- At least 4 GB of available RAM; a CUDA-enabled PyTorch installation is optional

## Windows and VS Code setup

1. Open this folder in VS Code and open **Terminal > New Terminal**.
2. Verify Python:

   ```powershell
   py --version
   ```

3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process
   Bypass` in that terminal, then activate again. In VS Code, choose
   `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

4. Install the pinned-compatible dependency ranges:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Launch the app from the project directory:

   ```powershell
   python -m streamlit run app.py
   ```

   Open the URL shown in the terminal, normally `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the `custom_style_gpt2_lab` folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select `app.py` as the
   **Main file path**.
3. Set the Python version to 3.10 or 3.11 and deploy.

The app downloads the base GPT-2 checkpoint on its first model operation and
saves fine-tuned checkpoints at runtime. The generated `outputs` directory and
local virtual environments are excluded from source control, so they do not
need to be uploaded. Streamlit Cloud instances need enough memory for the
GPT-2 model; use a short training run for the hosted demo.

## Dataset format

Upload a UTF-8 plain-text `.txt` file. Each non-empty line or paragraph can be
one example. The app normalizes whitespace, rejects very short input, tokenizes
the text into padded causal-language-model blocks, and uses an attention mask
so padding does not contribute to loss. `data\sample_dataset.txt` contains varied
campus announcements and is ready for the first demonstration.

## Demonstration workflow

1. Keep **Use the included sample campus-announcement dataset** selected, or
   upload another `.txt` dataset.
2. Set **Training steps** to 5-10, learning rate to `5e-5`, and generation
   length to 30-50 tokens.
3. Click **Fine-tune GPT-2**. The progress bar shows every optimizer update and
   the latest loss. Training is only reported as successful after the model and
   tokenizer checkpoint are saved and verified in `outputs\custom-gpt2`.
4. Inspect the final loss and line chart, then download `training_loss.csv`.
5. Enter a prompt such as `Campus Announcement: The technology club` and click
   **Generate and compare**.
6. Compare the same prompt's original GPT-2 and fine-tuned GPT-2 outputs. The
   fine-tuned model may reuse campus vocabulary, formal announcement phrasing,
   and event-oriented structure. A tiny dataset and short run can produce only
   subtle changes or can overfit repeated wording.

## Troubleshooting

- **Model download errors:** check internet access and retry. Hugging Face
  files are cached locally after a successful download.
- **Out of memory or slow CPU training:** use 1-5 steps, a shorter dataset, and
  a shorter generation length. Close other memory-heavy applications.
- **PowerShell activation error:** use
  `Set-ExecutionPolicy -Scope Process Bypass`, then activate `.venv` again.
- **No fine-tuned checkpoint:** generation is intentionally disabled until a
  complete successful training run saves `outputs\custom-gpt2`.
- **Port already in use:** run
  `python -m streamlit run app.py --server.port 8502`.
- **Cached stale code:** stop Streamlit with `Ctrl+C`, restart it, and reload
  the browser page.

No API key or paid service is required. Model training and generation happen
locally with PyTorch and Transformers.
