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

#データの統合
ID_list = []
mol_list =[]
for mol in train_mols:
    ID_list.append(mol.GetIntProp('ID'))
    mol_list.append(mol)
for mol in test_mols:
    ID_list.append(mol.GetIntProp('ID'))
    mol.SetProp('smiles', mol.GetProp('SMILES'))
    mol_list.append(mol)
    
mol_datasets = pd.DataFrame({'ID':ID_list,'mol':mol_list})

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

rm_num = 0
calc_num = 0
task_num = 0
#whileループ
while len(mol_datasets.index) > 0: 
    #forループ
    #コピーの作成
    loop_datasets = mol_datasets.copy()
    #for i, (ID, mol) in enumerate(zip(loop_datasets['ID'], loop_datasets['mol'])):
    for i, (ID, mol) in enumerate(zip(ID_list, mol_list)):
        try:
            smile = mol.GetProp('smiles')
            #logfile
            print('ID: {}, smiles: {}'.format(ID, smile))
            try:
                psi4.set_output_file("../../ForMolPredict/LogFiles/Solubility/{}.log".format(ID))
            except SystemError:
                print("rise system Error in line 73")
                #errorした分子の取得と削除
                mol_datasets.drop(ID-1, inplace = True)
                rm_num += 1
                task_num = data_num - rm_num - calc_num
                print('Remaining molecules:', task_num)
                psi4.core.opt_clean()
                print('skiped this molecule: ', smile)
                print('Error passed')
                break 
            #molオブジェクトに水素付加
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
            for atom in mol.GetAtoms():
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
                print("raise OptimizationConvergenceError in line 114")
                #errorした分子の取得と削除
                mol_datasets.drop(ID-1, inplace = True)
                rm_num += 1
                task_num = data_num - rm_num - calc_num
                print('Remaining molecules:', task_num)
                psi4.core.opt_clean()
                print('skiped this molecule: ', smile)
                print('Error passed')
                break
            except TimeoutError:
                print("raise TimeoutError in line 114")
                #errorした分子の取得と削除
                mol_datasets.drop(ID-1, inplace = True)
                rm_num += 1
                task_num = data_num - rm_num - calc_num
                print('Remaining molecules:', task_num)
                psi4.core.opt_clean()
                print('skiped this molecule: ', smile)
                print('Error passed')
                break
        except Exception as e:
            print('Unexpected Error')
            print(e)
            mol_datasets.drop(ID-1, inplace = True)
            rm_num += 1
            task_num = data_num - rm_num - calc_num
            print('Remaining molecules:', task_num)
            psi4.core.opt_clean()
            print('skiped this molecule: ', smile)
            print('Error passed')
            break
        except Exception:
            print('Rise Other Error')
            mol_datasets.drop(ID-1, inplace = True)
            rm_num += 1
            task_num = data_num - rm_num - calc_num
            print('Remaining molecules:', task_num)
            psi4.core.opt_clean()
            print('skiped this molecule: ', smile)
            print('Error passed')
            break
            
        #wavefunctionの書き出し
        wave_function.to_file("../../ForMolPredict/WaveFunctions/Solubility/{}_Sol_wavefunction".format(ID))
        #計算の終わった分子の削除
        mol_datasets.drop(ID-1, inplace = True)
        psi4.core.opt_clean()
        calc_num += 1
        task_num = data_num - rm_num - calc_num
        print("completed caluculations: ", calc_num)
        print('Remaining molecules:', task_num)
        
print("Calculation was finished!!")