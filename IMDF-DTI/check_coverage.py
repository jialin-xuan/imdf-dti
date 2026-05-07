
import joblib
import os

dataset = "Davis"
data_path = f"./DataSets/{dataset}.txt"
protein_path = f"./DataSets/Preprocessed/{dataset}-protein.pkl"
ligand_path = f"./DataSets/Preprocessed/{dataset}-ligand-hi.pkl"

print(f"Checking coverage for {dataset}...")

# Load raw data
with open(data_path, "r") as f:
    data_list = f.read().strip().split('\n')

raw_smiles = set([item.split(' ')[-3] for item in data_list])
raw_proteins = set([item.split(' ')[-2] for item in data_list])

print(f"Raw data: {len(raw_smiles)} unique ligands, {len(raw_proteins)} unique proteins.")

# Load preprocessed data
if os.path.exists(protein_path):
    protein_dict = joblib.load(protein_path)
    print(f"Loaded protein dict: {len(protein_dict)} entries.")
    missing_proteins = raw_proteins - set(protein_dict.keys())
    print(f"Missing proteins: {len(missing_proteins)}")
else:
    print("Protein pkl not found.")
    missing_proteins = raw_proteins

if os.path.exists(ligand_path):
    ligand_dict = joblib.load(ligand_path)
    print(f"Loaded ligand dict: {len(ligand_dict)} entries.")
    missing_ligands = raw_smiles - set(ligand_dict.keys())
    print(f"Missing ligands: {len(missing_ligands)}")
else:
    print("Ligand pkl not found.")
    missing_ligands = raw_smiles
