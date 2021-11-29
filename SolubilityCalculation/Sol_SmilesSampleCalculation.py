#rdkit関連のimport
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem import PandasTools

#psi4
import psi4

#その他のimport
import pandas as pd
import time
from timeout_decorator import timeout, TimeoutError

#分子の読み込み
suppl_train = Chem.SDMolSupplier('solubility.train.sdf')  #<-train
train_mols = [x for x in suppl_train]   #<-train

suppl_test = Chem.SDMolSupplier('solubility.test.sdf')  #<-test
test_mols = [x for x in suppl_test]   #<-test

#datasetの作成
ID_list = []
smiles_list = []
for mol in train_mols:
    ID_list.append(mol.GetIntProp('ID'))
    smiles_list.append(mol.GetProp('smiles'))
for mol in test_mols:
    ID_list.append(mol.GetIntProp('ID'))
    smiles_list.append(mol.GetProp('SMILES'))
    
mol_datasets = pd.DataFrame({'ID':ID_list,'smiles':smiles_list})

#Error分子
Error_mols = pd.read_table('Sol_ErrorMolecule.txt', sep = ' ')
Error_ID = Error_mols.ID
mol_datasets = mol_datasets.drop(Error_ID)
mol_datasets = mol_datasets.sort_values('ID')
mol_datasets = mol_datasets.reset_index(drop=True)

#データ数の取得
data_num = len(mol_datasets.index)
print('start with {} samples'.format(data_num))

"""Psi4"""
#スレッド数とメモリの設定
psi4.set_num_threads(nthread=5)
psi4.set_memory("10GB")

#wavefunction caluclation with timeout
timeout_sec = 60*60 #(sec)

@timeout(timeout_sec)
def opt(level, molecule):
    energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
    return energy, wave_function
    
    
