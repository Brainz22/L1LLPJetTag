# MyProject

## Project Structure
```bash
.
|-- data
|-- environment.yml
|-- pyproject.toml
|-- pytest.ini
|-- scripts
|   |-- run_analysis.py
|   |-- run_dataForge.py
|   `-- run_train.py
|-- src
|   `-- L1LLPJetTagger
|       |-- core.py
|       |-- dataForge.py
|       |-- explore_root.ipynb
|       |-- model
|       |-- plotting
|       |   `-- histo.py
|       |-- processor.py
|       |-- training
|       `-- utils
|           |-- config.py
|           |-- kinematics.py
|           `-- utils.py
`-- tests
    `-- test_core.py
```


Example Python project structure with CI and conda environment.

```bash
# Install micromamba (if not installed)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create and activate environment
micromamba create -f environment.yml
micromamba activate L1LLPJetTagger


# once per environment setup
pip install -e .
pip install pre-commit
pre-commit install

# now you can:
python scripts/run_analysis.py
pytest
