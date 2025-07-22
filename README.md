# MyProject

Example Python project structure with CI and conda environment.

```bash
# Install micromamba (if not installed)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create and activate environment
micromamba create -f environment.yml
micromamba activate sampleproject


# once per environment setup
pip install -e .
pip install pre-commit
pre-commit install

# now you can:
python scripts/run_analysis.py
pytest
