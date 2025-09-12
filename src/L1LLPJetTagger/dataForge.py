from L1LLPJetTagger import delta_phi
import numpy as np
from L1LLPJetTagger.utils.utils import one_hot_encode_pdgid
from L1LLPJetTagger.utils.utils import pad_and_fill
from L1LLPJetTagger.utils.utils import known_ids
from L1LLPJetTagger.utils.config import config
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import awkward as ak
import h5py
import os

DEBUG = False


def debug_print(events, num_jets=5):

    print("\nLoaded fields:")
    for field in events.fields:
        print(f" - {field}: {ak.type(events[field])}")

    print("\nFirst {} jets:".format(num_jets))
    for i in range(min(num_jets, len(events["jet_pt"]))):
        pt = ak.to_list(events["jet_pt"][i])
        eta = ak.to_list(events["jet_eta"][i])
        phi = ak.to_list(events["jet_phi"][i])
        pid = ak.to_list(events["jet_pfcand_id"][i])
        cand_pt = ak.to_list(events["jet_pfcand_pt_phys"][i])
        cand_eta = ak.to_list(events["jet_pfcand_eta_phys"][i])
        cand_phi = ak.to_list(events["jet_pfcand_phi_phys"][i])
        vx = ak.to_list(events["jet_pfcand_track_vx"][i])
        vy = ak.to_list(events["jet_pfcand_track_vy"][i])
        vz = ak.to_list(events["jet_pfcand_track_vz"][i])
        dxy = ak.to_list(events["jet_pfcand_dxy"][i])

        print(f"\nJet {i}:")
        print(f"  pt:  {pt:.2f}")
        print(f"  eta: {eta:.2f}")
        print(f"  phi: {phi:.2f}")
        print(f"  nPFCand: {len(vx)}")
        print(f"  pfCand_ID: {pid[:]}")
        print(f"  pfCand_pt: {cand_pt[:]}")
        print(f"  pfCand_eta: {cand_eta[:]}")
        print(f"  pfCand_phi: {cand_phi[:]}")
        print(f"  track_vx: {vx[:]}")
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
    data_type: str = "UNKNOWN",
    num_constituents: int = 10,
    train_split: float = 0.8,
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

    # Debug print to show first N jets
    if DEBUG:
        debug_print(events, 2)

    # Pad to fixed length num_constituents per jet using 0 (NJets, NConstituents)
    pid = pad_and_fill(events["jet_pfcand_id"], num_constituents)
    pt = pad_and_fill(events["jet_pfcand_pt_phys"], num_constituents)
    eta = pad_and_fill(events["jet_pfcand_eta_phys"], num_constituents)
    phi = pad_and_fill(events["jet_pfcand_phi_phys"], num_constituents)
    vx = pad_and_fill(events["jet_pfcand_track_vx"], num_constituents)
    vy = pad_and_fill(events["jet_pfcand_track_vy"], num_constituents)
    vz = pad_and_fill(events["jet_pfcand_track_vz"], num_constituents)

    # keep track of dummy entries
    mask = np.abs(pt) > 1e-6

    # Normalize to jet-level quantities
    rel_pt = pt / events["jet_pt_phys"][..., None]
    rel_eta = eta - events["jet_eta_phys"][..., None]
    rel_phi = delta_phi(phi, events["jet_phi_phys"][..., None])
    rel_eta = ak.where(mask, rel_eta, 0.0)
    rel_phi = ak.where(mask, rel_phi, 0.0)

    # massage the pid array to be flat and do the one-hot encoding
    flat_pid = ak.to_numpy(pid).reshape(-1)
    one_hot = one_hot_encode_pdgid(flat_pid)
    one_hot = one_hot.reshape(len(pid), num_constituents, len(known_ids))

    # shape: (n_jets, n_cands,n_features)
    constituents = ak.concatenate(
        [
            one_hot,
            rel_pt[..., None],
            rel_eta[..., None],
            rel_phi[..., None],
            vx[..., None],
            vy[..., None],
            vz[..., None],
        ],
        axis=-1,
    )
    # Reshape to (n_jets, n_cands * n_features)
    flat = ak.to_numpy(constituents).reshape(len(vx), -1)

    # split dataset into a training and a testing set randomly
    # Shuffle the data for randomness
    n_total = flat.shape[0]
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Split into training and testing sets
    train_size = int(train_split * n_total)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = flat[train_indices]
    test_data = flat[test_indices]

    print(f"\nFinal array shape before splitting: {flat.shape}")
    print(f"Final array shape for test data: {test_data.shape}")
    print(f"Final array shape for train data: {train_data.shape}")
    print(f"Total jets: {flat.shape[0]}")

    # Save to HDF5
    out_path_h5 = os.path.join(out_path, data_type + "_jet_data.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_data", data=flat)

    out_path_h5 = os.path.join(out_path, data_type + "_train.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_constituents", data=train_data)

    out_path_h5 = os.path.join(out_path, data_type + "_test.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_constituents", data=test_data)

    print(f"\nSaved data to {out_path}")

    return events
