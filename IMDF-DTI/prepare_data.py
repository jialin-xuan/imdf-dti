
import joblib
import os
import sys
import torch
from tqdm import tqdm

# Add current directory to path so we can import utils
sys.path.append(os.getcwd())

from utils.protein_init import protein_init
from utils.ligand_init import ligand_init

dataset = "Davis"
data_path = f"./DataSets/{dataset}.txt"
protein_path_old = f"./DataSets/Preprocessed/{dataset}-protein.pkl"
protein_path_new = f"./DataSets/Preprocessed/{dataset}-protein-new.pkl"
ligand_path_new = f"./DataSets/Preprocessed/{dataset}-ligand-hi-new.pkl"

print(f"Preparing data for {dataset}...")

# Load raw data
with open(data_path, "r") as f:
    data_list = f.read().strip().split('\n')

# Extract unique sequences and SMILES
raw_proteins = list(set([item.split(' ')[-2] for item in data_list]))
raw_smiles = list(set([item.split(' ')[-3] for item in data_list]))

print(f"Raw data: {len(raw_smiles)} unique ligands, {len(raw_proteins)} unique proteins.")

# --- Process Proteins ---
existing_proteins = {}
if os.path.exists(protein_path_new):
    print(f"Loading existing new protein dict from {protein_path_new}")
    existing_proteins = joblib.load(protein_path_new)
elif os.path.exists(protein_path_old):
    print(f"Loading existing old protein dict from {protein_path_old}")
    existing_proteins = joblib.load(protein_path_old)

# Identify missing proteins
missing_proteins_seqs = [seq for seq in raw_proteins if seq not in existing_proteins]
print(f"Found {len(existing_proteins)} existing proteins. Missing: {len(missing_proteins_seqs)}")

if missing_proteins_seqs:
    print("Generating features for missing proteins... This may take a while.")
    # Call protein_init for missing sequences
    # protein_init returns a dict {seq: features}
    new_protein_dict = protein_init(missing_proteins_seqs)
    
    # Merge
    existing_proteins.update(new_protein_dict)
    
    # Save
    print(f"Saving merged protein dict to {protein_path_new}")
    joblib.dump(existing_proteins, protein_path_new)
else:
    print("All proteins are already processed.")
    if not os.path.exists(protein_path_new) and existing_proteins:
        print(f"Saving existing protein dict to {protein_path_new}")
        joblib.dump(existing_proteins, protein_path_new)

# --- Process Ligands ---
# Ligand init is fast, so we can just regenerate or update.
# RunModel.py does: ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
print("Generating ligand features...")
ligand_dict = ligand_init(raw_smiles, mode='BRICS')
print(f"Saving ligand dict to {ligand_path_new}")
joblib.dump(ligand_dict, ligand_path_new)

print("Data preparation complete.")
