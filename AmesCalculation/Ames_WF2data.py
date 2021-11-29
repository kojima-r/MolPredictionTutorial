#rdkit関連のimport
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule

#psi4
import psi4

#その他のimport
import pandas as pd
import numpy as np
import glob
import re

files = []
for file in glob.glob("../WaveFunctions/Ames/*_Ames_wavefunction.npy"):
    files.append(file)

data_num = len(files)
print("start with {} wavefunctions".format(data_num))

#wavefunctionの取り出し

#スレッド数とメモリの設定
psi4.set_num_threads(nthread=5)
psi4.set_memory("10GB")

ID_list = []
wavefunction_list = []
for file in files:
    pre_ID = re.findall('../WaveFunctions/Ames/(.*)_Ames_wavefunction.npy', file)
    ID = pre_ID[0]
    ID_list.append(int(ID))
    wave_function = psi4.core.Wavefunction.from_file(file)
    wavefunction_list.append(wave_function) 
    
#データのリスト化とソート
data_sets = pd.DataFrame({'ID': ID_list, 'wavefunction': wavefunction_list})
data_sets = data_sets.sort_values('ID')
sort_data_sets = data_sets.reset_index(drop=True)

#元データの読み込み
original_df = pd.read_csv( 'AvailableAmesDataSets.csv')
print('original data has {} molecules'.format(len(original_df.index))) #6506

#smiles & activityの取得
smiles_list = []
act_list = []
for ID in sort_data_sets.ID :
    same_df = original_df.query('ID == @ID')
    smile = same_df['smiles'].values[0]
    act = same_df['activity'].values[0]
    smiles_list.append(smile)
    act_list.append(float(act))

sort_data_sets['smiles'] = smiles_list
sort_data_sets['activity'] = act_list

#data listの作成
energy_list = []
homo_list = []
lumo_list = []
HLgap_list = []
charge_ave_list = []
charge_var_list = []
dipole_list = []
index_list = []
mass_list = []
vol_list = []
dens_list = []
for wave_function,ID,smile in zip(sort_data_sets.wavefunction,sort_data_sets.ID,sort_data_sets.smiles):
    """rdkit"""
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))   #rdkitのmolオブジェクトの生成
    AllChem.EmbedMolecule(rdmol)     #埋め込み(3D)
    vol = AllChem.ComputeMolVolume(rdmol)   #体積の計算
    vol_list.append(vol)
    """psi4"""
    #energy
    energy = wave_function.energy()
    energy_list.append(energy)
    #homo/lumo
    LUMO_idx = wave_function.nalpha()
    HOMO_idx = LUMO_idx - 1
    homo =wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]
    lumo = wave_function.epsilon_a_subset("AO", "ALL").np[LUMO_idx]
    homo_list.append(homo)
    lumo_list.append(lumo)
    HLgap = lumo - homo
    HLgap_list.append(HLgap)
    #MULLIKEN電荷
    psi4.oeprop(wave_function, "MULLIKEN_CHARGES")
    charges = np.array(wave_function.atomic_point_charges())
    charge_ave_list.append(np.average(charges))
    charge_var_list.append(np.var(charges))
    #dipole
    dipole_x, dipole_y, dipole_z = wave_function.variable('SCF DIPOLE X'), wave_function.variable('SCF DIPOLE Y'), wave_function.variable('SCF DIPOLE Z')
    dipole_moment = np.sqrt(dipole_x ** 2 + dipole_y ** 2 + dipole_z ** 2)
    dipole_list.append(dipole_moment)
    #atomごとの取り扱い
    mol = wave_function.molecule() #molオブジェクトの生成
    atom_num = mol.natom() #atom数
    #mass
    atom_mass_list = []
    for n in range(0,atom_num):
        mass = mol.mass(n)
        #print(n, mass)
        atom_mass_list.append(mass)
    mol_mass = sum(atom_mass_list)
    mass_list.append(mol_mass)
    atom_mass_list.clear()
    #density
    dens = mol_mass/vol
    dens_list.append(dens)
#data_list
sort_data_sets['energy'] = energy_list
sort_data_sets['homo'] = homo_list
sort_data_sets['lumo'] = lumo_list
sort_data_sets['HLgap'] = HLgap_list
sort_data_sets['charge_ave'] = charge_ave_list
sort_data_sets['charge_var'] = charge_var_list
sort_data_sets['dipole'] = dipole_list
sort_data_sets['mass'] = mass_list
sort_data_sets['vol'] = vol_list
sort_data_sets['dens'] = dens_list

print_data_sets = sort_data_sets.drop(columns='wavefunction')
print_data_sets.to_csv("Ames_{}samples.csv".format(data_num),index=False)