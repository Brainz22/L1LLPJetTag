from L1LLPJetTagger.utils.utils import one_hot_encode_pdgid
from L1LLPJetTagger.utils.utils import pad_and_fill
from L1LLPJetTagger.utils.utils import known_ids
from L1LLPJetTagger.utils.config import config
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import awkward as ak
import h5py
import os


def debug_print(events):

    print("\nLoaded fields:")
    for field in events.fields:
        print(f" - {field}: {ak.type(events[field])}")

    print("\nFirst 5 jets:")
    for i in range(min(5, len(events["jet_pt"]))):
        pt = ak.to_list(events["jet_pt"][i])
        eta = ak.to_list(events["jet_eta"][i])
        phi = ak.to_list(events["jet_phi"][i])
        pid = ak.to_list(events["jet_pfcand_id"][i])
        vx = ak.to_list(events["jet_pfcand_track_vx"][i])
        vy = ak.to_list(events["jet_pfcand_track_vy"][i])
        vz = ak.to_list(events["jet_pfcand_track_vz"][i])
        dxy = ak.to_list(events["jet_pfcand_dxy"][i])

        print(f"\nJet {i}:")
        print(f"  pt:  {pt:.2f}")
        print(f"  eta: {eta:.2f}")
        print(f"  phi: {phi:.2f}")
        print(f"  nPFCand: {len(vx)}")
        print(f"  track_vx: {vx[:]}")
        print(f"  pfCand_ID: {pid[:]}")
        print(f"  track_vy: {vy[:]}")
        print(f"  track_vz: {vz[:]}")
        print(f"  track_dxy: {dxy[:]}")

    print("\nConfig paths:")
    print(f"  pkgdir: {config.PKG_ROOT}")
    print(f"  projdir: {config.PROJECT_ROOT}")


def forge_h5(
    root_path: str,
    out_path: str,
    tree_name: str,
    num_constituents: int = 10,
):
    """
    Processes a ROOT file containing jet and constituent information and saves the output in HDF5 format.

    This function loads jet-level constituent data from a ROOT TTree, pads each jet
    to a fixed number of constituents, and stores the processed data as a flat array in an HDF5 file.

    Args:
        root_path (str): Path to the input ROOT file.
        out_path (str): Path to the output HDF5 file.
        tree_name (str): Name of the TTree in the ROOT file to process.

    Returns:
        ak.Array: Awkward Array of events loaded from the ROOT file using BaseSchema.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load NanoEvents using BaseSchema for custom structure
    events = NanoEventsFactory.from_root(
        f"{root_path}:{tree_name}", schemaclass=BaseSchema
    ).events()

    # Debug print to show first 5 jets
    debug_print(events)

    # Pad to fixed length num_constituents per jet using 0 (NJets, NConstituents)
    pid = pad_and_fill(events["jet_pfcand_id"], num_constituents)
    vx = pad_and_fill(events["jet_pfcand_track_vx"], num_constituents)
    vy = pad_and_fill(events["jet_pfcand_track_vy"], num_constituents)
    vz = pad_and_fill(events["jet_pfcand_track_vz"], num_constituents)

    # massage the pid array to be flat and do the one-hot encoding
    flat_pid = ak.to_numpy(pid).reshape(-1)
    one_hot = one_hot_encode_pdgid(flat_pid)
    one_hot = one_hot.reshape(len(pid), num_constituents, len(known_ids))

    # shape: (n_jets, n_cands,n_features)
    constituents = ak.concatenate(
        [one_hot, vx[..., None], vy[..., None], vz[..., None]], axis=-1
    )
    # Reshape to (n_jets, n_cands * n_features)
    flat = ak.to_numpy(constituents).reshape(len(vx), -1)

    # Save to HDF5
    with h5py.File(out_path, "w") as f:
        f.create_dataset("jet_constituents", data=flat)

    print(f"\nSaved {flat.shape[0]} jets to {out_path}")

    return events
