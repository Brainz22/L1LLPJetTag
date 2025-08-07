from pathlib import Path


class _Config:
    @property
    def PROJECT_ROOT(self):
        return Path(__file__).resolve().parent.parent.parent.parent

    @property
    def PKG_ROOT(self):
        return Path(__file__).resolve().parent.parent

    @property
    def N_PART_PER_JET(self):
        return 10

    @property
    def N_FEAT(self):
        return 14

    def __setattr__(self, name, value):
        raise AttributeError(f"Cannot modify constant '{name}'")


config = _Config()
