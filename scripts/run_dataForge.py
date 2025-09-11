from L1LLPJetTagger.dataForge import forge_h5
from L1LLPJetTagger.utils.config import config

import os

if __name__ == "__main__":
    root_path = os.path.join(config.PROJECT_ROOT, "DY_1k.root")
    # root_path = os.path.join(config.PROJECT_ROOT, "jetTuple_extended_5.root")
    out_path = os.path.join(config.PROJECT_ROOT, "data/")
    tree_name = "outnano/Jets"
    data_type = "DY"
    # data_type = "Sig"

    events = forge_h5(
        root_path=root_path, out_path=out_path, tree_name=tree_name, data_type=data_type
    )
