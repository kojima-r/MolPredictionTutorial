#rdkit関連のimport
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools, Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit.Chem.Draw import IPythonConsole

#psi4
import psi4

#その他のimport
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
from timeout_decorator import timeout, TimeoutError
import copy

#分子の読み込み
suppl_train = Chem.SDMolSupplier('./sdf_data/solubility.train.sdf')  #<-train
train_mols = [x for x in suppl_train]   #<-train

#suppl_test = Chem.SDMolSupplier('./sdf_data/solubility.test.sdf')  #<-test
#test_mols = [x for x in suppl_test]   #<-test

"""Psi4"""
#スレッド数とメモリの設定
psi4.set_num_threads(nthread=5)
psi4.set_memory("10GB")

#wavefunction caluclation with timeout
timeout_sec = 30*60 #(sec)

@timeout(timeout_sec)
def opt(level, molecule):
    energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
    return energy, wave_function

#データの複製(train or test)
mols = copy.copy(train_mols)  #<-train
#mols = copy.copy(test_mols)  #<-test
#データ数の取得
data_num = len(mols)
print('start with {} moleculs'.format(data_num))

#リスト
opt_error_mols = []
timeout_error_mols = []
unexpected_error_mols = []
rm_num = 0
calc_num = 0
unexpexcted_num = 0
#whileループ
while len(mols)> 0: 
    #forループ
    #コピーの作成
    mols_for = copy.copy(mols)
    for i, mol in enumerate(mols_for):
        smile = mol.GetProp('smiles')  #<-train
        #smile = mol.GetProp('SMILES')  #<-test
        print(smile)
        if smile == "CNC(NCCSCc1ncnc1C)=NC#N" or smile == "C1C(=O)C=C2CCC3C4CCC(O)(C#C)C4(C)CCC3C2(C)C1" or smile =="NC(=NC(#N))N" or smile == "N#CCCCl":
            mols.remove(mol)
            rm_num += 1
            print('total molecules:', data_num - rm_num)
            psi4.core.opt_clean()
            print('skiped this molecule: ', mol)
            break 
        try:
            #logfile
            try:
                psi4.set_output_file("{}.log".format(smile))
            except SystemError:
                print("rise system Error in line 78", "skip this molecule")
                #errorした分子の取得と削除
                opt_error_mols.append([mol.GetProp('ID'), mol])
                mols.remove(mol)
                rm_num += 1
                print('total molecules:', data_num - rm_num)
                psi4.core.opt_clean()
                print('Error passed')
                break
            #smilesから三次元構造を作る→雑に構造最適化
            Hmol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomseed = 1
            EmbedMolecule(Hmol, params)
            #MMFF（Merck Molecular Force Field） で構造最適化
            MMFFOptimizeMolecule(Hmol)
            conf = Hmol.GetConformer()
            #プラスとマイナスの数を数える
            plus = smile.count('+')
            minus = smile.count('-')
            #電荷の計算
            charge = plus - minus + 0
            #Psi4に入力可能な形式に変換
            #電荷とスピン多重度の設定
            mol_input = str(charge) + " 1"
            print(str(i+1) + ": " + mol_input + ": "+ smile)
            #各々の原子の座標をXYZで記述
            for atom in Hmol.GetAtoms():
                mol_input += "\n" + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x)\
                + " " + str(conf.GetAtomPosition(atom.GetIdx()).y)\
                + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)
            molecule = psi4.geometry(mol_input)
            #汎関数、基底関数の指定
            level = "HF/sto-3g"
            #構造最適化計算の実行
            psi4.set_options({'geom_maxiter': 1000})
            try:
                energy, wave_function = opt(level, molecule)
            except psi4.OptimizationConvergenceError:
                print("raise OptimizationConvergenceError in line 123, skip optimazation: " + smile)
                #errorした分子の取得と削除
                opt_error_mols.append([mol.GetProp('ID'), mol])
                mols.remove(mol)
                rm_num += 1
                print('total molecules:', data_num - rm_num)
                psi4.core.opt_clean()
                print('Error passed')
                break
            except TimeoutError:
                print("raise TimeoutError in line 123, skip optimazation: " + smile)
                #errorした分子の取得と削除
                timeout_error_mols.append([mol.GetProp('ID'), mol])
                mols.remove(mol)
                rm_num += 1
                print('total molecules:', data_num - rm_num)
                psi4.core.opt_clean()
                print('Error passed')
                break
        except Exception as e:
            print('Unexpected Error')
            print(e)
            unexpected_num += 1
            #print("unexpected_num", unexpected_num)
            if unexpected_num == 2:
                mols.remove(mol)
                rm_num += 1
                print('Rise unexpected error twice')
                print('total molecules:', data_num - rm_num)
                psi4.core.opt_clean()
                print('Error passed')
                unexpected_num = 0
                break
            unexpected_error_mols.append([mol.GetProp('ID'), mol])
            print('caluclation continue')
            print('total molecules:', data_num - rm_num)
            print('Error through')
            break
        except:
            print('rise other Error')
            mols.remove(mol)
            rm_num += 1
            print('total molecules:', data_num - rm_num)
            psi4.core.opt_clean()
            print('Error passed')
            break
        #wavefunctionの書き出し
        wave_function.to_file("{}_{}_train_wavefunction".format(mol.GetProp('ID'), smile)) #<-train
        #wave_function.to_file("{}_{}_test_wavefunction".format(mol.GetProp('ID'), smile))  #<-test
        #計算の終わった分子の削除
        mols.remove(mol)
        calc_num += 1
        print("completed caluculations: ", calc_num)
        unexpected_num = 0
        
print("calculation was finished!!")
print('total molecules:', data_num - rm_num)
#opt_error_df = pd.DataFrame(opt_error_smiles)
#time_error_df = pd.DataFrame(time_error_smiles)
print("Opt Error mols")
print(opt_error_mols)
print("Timeout Error mols")
print(timeout_error_mols)
print("Unexpected Error mols")
print(unexpected_error_mols)