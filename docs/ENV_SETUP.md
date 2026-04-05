# Environment Setup

This project requires Python 3.12. Two setup workflows are supported.

## Option 1: Conda (recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:

```bash
conda env create -f environment.yml
conda activate cascading-disaster
```

3. (Optional) Register a Jupyter kernel:

```bash
python -m ipykernel install --user --name=cascading-disaster --display-name "Cascading Disaster (py3.12)"
```

## Option 2: pip / venv

1. Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install the project in editable mode:

```bash
pip install --upgrade pip
pip install -e .
pip install ipykernel nbformat notebook
python -m ipykernel install --user --name=cascading-disaster-py --display-name "Cascading Disaster (venv)"
```

## Notes

- If you prefer JupyterLab, replace `notebook` with `jupyterlab` in the commands above.
- The `environment.yml` uses the `conda-forge` channel. If you hit resolution issues, try `conda update -n base -c defaults conda`.
- XGBoost and additional ML libraries will be pulled in by `pyproject.toml` dependencies automatically.
