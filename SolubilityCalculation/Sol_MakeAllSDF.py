#rdkit関連のimport
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

#psi4
import psi4

#その他のimport
import pandas as pd
import numpy as np
import glob
import re


def MakeMolData(wf,smile,SOL,SOL_class):
    mol_features = []
    mol_features_name = []
    atom_features = []
    atom_features_name = []
    mol_features.append(SOL)
    mol_features_name.append('SOL')
    mol_features.append(SOL_class)
    mol_features_name.append('SOL_class')
    """rdkit"""
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))   #rdkitのmolオブジェクトの生成
    AllChem.EmbedMolecule(rdmol)     #埋め込み(3D)
    vol = AllChem.ComputeMolVolume(rdmol)   #体積の計算
    mol_features.append(vol)
    mol_features_name.append('Volume')
    """psi4"""
    #energy
    energy = wf.energy()
    mol_features.append(energy)
    mol_features_name.append('Energy')
    #homo/lumo
    LUMO_idx = wf.nalpha()
    HOMO_idx = LUMO_idx - 1
    homo =wf.epsilon_a_subset("AO", "ALL").np[HOMO_idx]
    lumo = wf.epsilon_a_subset("AO", "ALL").np[LUMO_idx]
    mol_features.append(homo)
    mol_features_name.append('HOMO')
    mol_features.append(lumo)
    mol_features_name.append('LUMO')
    HLgap = lumo - homo
    mol_features.append(HLgap)
    mol_features_name.append('HLgap')
    #MULLIKEN電荷
    psi4.oeprop(wf, "MULLIKEN_CHARGES")
    Mcharges = np.array(wf.atomic_point_charges())
    mol_features.append(np.average(Mcharges))
    mol_features_name.append('Mcharge_ave')
    mol_features.append(np.var(Mcharges))
    mol_features_name.append('Mcharge_var')
    atom_features.append(Mcharges.tolist())
    atom_features_name.append('Mcharges')
    #Löwdin電荷
    psi4.oeprop(wf, "LOWDIN_CHARGES")
    Lcharges = np.array(wf.atomic_point_charges())
    mol_features.append(np.average(Lcharges))
    mol_features_name.append('Lcharge_ave')
    mol_features.append(np.var(Lcharges))
    mol_features_name.append('Lcharge_var')
    atom_features.append(Lcharges.tolist())
    atom_features_name.append('Lcharges')
    #dipole
    dipole_x, dipole_y, dipole_z = wf.variable('SCF DIPOLE X'), wf.variable('SCF DIPOLE Y'), wf.variable('SCF DIPOLE Z')
    dipole_moment = np.sqrt(dipole_x ** 2 + dipole_y ** 2 + dipole_z ** 2)
    mol_features.append(dipole_moment)
    mol_features_name.append('dipole')
    #atomごとの取り扱い
    mol = wf.molecule() #molオブジェクトの生成
    atom_num = mol.natom() #atom数
    mol_features.append(atom_num)
    mol_features_name.append('Atom_num')
    #mass
    atom_mass_list = []
    x_dem_list = []
    y_dem_list = []
    z_dem_list = []
    symbol_list = []
    for n in range(atom_num):
        mass = mol.mass(n)
        atom_mass_list.append(mass)
        xyz = mol.xyz(n)
        x_dem_list.append(xyz[0])
        y_dem_list.append(xyz[1])
        z_dem_list.append(xyz[2])
        symbol_list.append(mol.symbol(n))
    atom_features.append(atom_mass_list)
    atom_features_name.append('Mass')
    mol_mass = sum(atom_mass_list)
    mol_features.append(mol_mass)
    mol_features_name.append('Mass')
    atom_features.append(x_dem_list)
    atom_features_name.append('X_dem')
    atom_features.append(y_dem_list)
    atom_features_name.append('Y_dem')
    atom_features.append(z_dem_list)
    atom_features_name.append('Z_dem')
    atom_features.append(symbol_list)
    atom_features_name.append('Symbol')
    #density
    dens = mol_mass/vol
    mol_features.append(dens)
    mol_features_name.append('Density')
    
    return mol_features, mol_features_name, atom_features, atom_features_name


def MakeAllSDF(ID,wf,smile,SOL,SOL_class,writer):
    mol = Chem.MolFromSmiles(smile)
    #smilesから三次元構造を作る→雑に構造最適化
    Hmol = Chem.AddHs(mol)
    params = ETKDGv3()
    params.randomseed = 1
    EmbedMolecule(Hmol, params)
    #MMFF（Merck Molecular Force Field） で構造最適化
    MMFFOptimizeMolecule(Hmol)
    mol_features, mol_features_name, atom_features, atom_features_name = MakeMolData(wf,smile,SOL,SOL_class)
    #molecule
    for mol_feature, mol_festure_name in zip(mol_features, mol_features_name):
        if mol_festure_name == 'SOL_class':
            Hmol.SetProp(mol_festure_name,mol_feature)
        else:
            Hmol.SetDoubleProp(mol_festure_name,mol_feature)
    #atom
    for natom in range(Hmol.GetNumAtoms()):
        atom = Hmol.GetAtomWithIdx(natom)
        for atom_feature,atom_feature_name in zip(atom_features, atom_features_name):
            if atom_feature_name == 'Symbol':
                atom.SetProp(atom_feature_name,atom_feature[natom-1])
            else :
                atom.SetDoubleProp(atom_feature_name,atom_feature[natom-1])
        for atom_feature_name in atom_features_name:
            if atom_feature_name == 'Symbol':
                Chem.CreateAtomStringPropertyList(Hmol, atom_feature_name)
            else :
                Chem.CreateAtomDoublePropertyList(Hmol, atom_feature_name)
    writer.write(Hmol)

    
files = []
for file in glob.glob("../../ForMolPredict/WaveFunctions/Solubility_from_smiles/*_Sol_wavefunction.npy"):
    files.append(file)

#wavefunctionの取り出し

#スレッド数とメモリの設定
psi4.set_num_threads(nthread=5)
psi4.set_memory("10GB")

#データの編集
last_data_ID = 0

ID_list = []
wavefunction_list = []
for file in files:
    pre_ID = re.findall('../../ForMolPredict/WaveFunctions/Solubility_from_smiles/(.*)_Sol_wavefunction.npy', file)
    ID = int(pre_ID[0])
    if ID > last_data_ID:
        ID_list.append(ID)
        wave_function = psi4.core.Wavefunction.from_file(file)
        wavefunction_list.append(wave_function)
    else :
        continue
    
#データのリスト化とソート
data_sets = pd.DataFrame({'ID': ID_list, 'wavefunction': wavefunction_list})
data_sets = data_sets.sort_values('ID')
sort_data_sets = data_sets.reset_index(drop=True)

data_num = len(sort_data_sets.index)
print("start with {} wavefunctions".format(data_num))

#元データの読み込み
original_df = pd.read_csv( 'AvailableSolubilityDataSets.csv')
print('original data has {} molecules'.format(len(original_df.index))) 

#smiles & SOLdataの取得
smiles_list = []
SOL_list = []
SOL_class_list = []
drop_ID_list =[]
for ID in sort_data_sets.ID :
    same_df = original_df[original_df['ID']==ID]
    if(len(same_df.ID)==0):
        drop_ID_list.append(ID)
        print('skiped {} molecule'.format(ID))
        continue
    print('sameID{}=ID{}'.format(same_df['ID'].values[0],ID))
    print('ID:{}'.format(ID),':',same_df['smiles'].values[0])
    smile = same_df['smiles'].values[0]
    SOL = same_df['SOL'].values[0]
    SOL_class = same_df['SOL_Class'].values[0]
    smiles_list.append(smile)
    SOL_list.append(float(SOL))
    SOL_class_list.append(SOL_class)

for drop_ID in drop_ID_list:
    sort_data_sets.drop(sort_data_sets[sort_data_sets.ID == drop_ID].index,inplace = True)
sort_data_sets['smiles'] = smiles_list
sort_data_sets['SOL'] = SOL_list
sort_data_sets['SOL_class'] = SOL_class_list

writer=Chem.SDWriter('../../ForMolPredict/SDF_files/SOL/SOL_AllMOL.sdf')
for index, row in sort_data_sets.iterrows():
    ID = row['ID']
    smile = row['smiles']
    wf = row['wavefunction']
    SOL = row['SOL']
    SOL_class = row['SOL_class']   
    try: 
        MakeAllSDF(ID,wf,smile,SOL,SOL_class,writer)
    except :
        print('Rise an error. skip this molecule ID: {}'.format(ID))
        continue
    print('Molecule(ID={}) was added!'.format(ID))
writer.close()
print('completed!!!')