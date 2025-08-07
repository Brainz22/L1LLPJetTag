from pathlib import Path


class _Config:
    @property
    def PROJECT_ROOT(self):
        return Path(__file__).resolve().parent.parent.parent.parent

    @property
    def PKG_ROOT(self):
        # Go from src/L1LLPJetTagger/config.py → src/L1LLPJetTagger
        return Path(__file__).resolve().parent.parent

    def __setattr__(self, name, value):
        raise AttributeError(f"Cannot modify constant '{name}'")


config = _Config()
