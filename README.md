# Swarm GPU-X — Streamlit Edition

Lightweight Streamlit port of the provided React GPU backtester. It implements the same procedural agent metadata, CPU fallback signals and equity simulation. Optional GPU support via CuPy may be enabled if you install it separately.

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (GPU): install CuPy for your CUDA / platform if you want GPU acceleration (not required):

```bash
# Example for CUDA 11.8 (check CuPy docs for the correct package for your GPU/driver)
pip install cupy-cuda118
```

## Run

```bash
streamlit run streamlit_app.py
```

## Usage

- Upload a CSV (first header line is used for delimiter detection). The app expects a date column then a close column (common close positions: column 2,5,6 — it will try to auto-detect).
- Click `Execute Backtest` to run the swarm of agents. The app shows top equity curves, rankings and processing metrics.

## Notes

- This implementation focuses on parity with the original JS logic and provides a clear CPU fallback path. A full GPU kernel port would require re-implementing the compute kernels with CuPy/Numba and is left as an enhancement.

## Publish to GitHub

1. Create a new repository on GitHub (private or public).
2. From your project folder, initialize git and push:

```bash
git init
git add .
git commit -m "Add Streamlit Swarm GPU-X app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Replace `<your-username>` and `<repo-name>` with your values.

## Deploy to Streamlit Cloud (share.streamlit.io)

1. Sign in to Streamlit Community Cloud and create a new app.
2. Connect the GitHub repo created above and select the branch and `streamlit_app.py` as the entrypoint.
3. Provide the required environment (you may add a `requirements.txt` — already included).
4. Deploy — Streamlit Cloud will build and run the app.

Note: For GPU/CuPy support you will need a custom deployment target that provides CUDA; Streamlit Community Cloud does not provide GPUs by default.

### Quick push (helper)

If you'd like a convenience helper, the repository includes `scripts/push_to_github.sh` which will attempt to use the `gh` CLI to create a repository and push the code. Usage:

```bash
chmod +x scripts/push_to_github.sh
./scripts/push_to_github.sh
```

If you don't have `gh`, run the manual `git` commands above after creating a repo on GitHub.

### Continuous Integration

A basic GitHub Actions workflow is included at `.github/workflows/ci.yml` that runs on pushes and pull requests to `main`. It installs dependencies, runs a syntax compile check and does a smoke import of the app to catch syntax errors early.
