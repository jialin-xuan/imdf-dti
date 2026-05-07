
import joblib
import os

try:
    p = joblib.load('./DataSets/Preprocessed/Davis-protein-new.pkl')
    l = joblib.load('./DataSets/Preprocessed/Davis-ligand-hi-new.pkl')
    print(f"Proteins: {len(p)}")
    print(f"Ligands: {len(l)}")
    
    expected_proteins = 379
    expected_ligands = 68
    
    if len(p) >= expected_proteins and len(l) >= expected_ligands:
        print("Data verification passed!")
    else:
        print(f"Data incomplete. Expected {expected_proteins} proteins and {expected_ligands} ligands.")
except Exception as e:
    print(f"Error loading data: {e}")
