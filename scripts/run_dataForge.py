from L1LLPJetTagger.dataForge import forge_h5

if __name__ == "__main__":
    root_path = "../jetTuple_extended_5.root"
    out_path = "../data/jet_data.h5"
    tree_name = "outnano/Jets"

    events = forge_h5(root_path=root_path, out_path=out_path, tree_name=tree_name)
