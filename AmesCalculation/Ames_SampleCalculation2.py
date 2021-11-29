#rdkit関連のimport
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

#psi4
import psi4

#その他のimport
import pandas as pd
import time
from timeout_decorator import timeout, TimeoutError

#pandasのデータフレームに分子を読み込み
original_dataset = pd.read_csv( "AvailableAmesDataSets.csv")

#smilesからrdKitのMolオブジェクトの構築
PandasTools.AddMoleculeColumnToFrame(frame=original_dataset, smilesCol = 'smiles')

#読み込めない分子を削除
original_dataset[ 'MOL'] = original_dataset.ROMol.map(lambda x: False if x == None else True)
del_index = original_dataset[original_dataset.MOL == False].index
edited_dataset = original_dataset.drop(del_index)

#計算済みのwavefunctionの確認
files = []
for file in glob.glob("../WaveFunctions/Ames/*_Ames_wavefunction.npy"):
    files.append(file)
    
ID_list = []
for file in files:
    pre_ID = re.findall('../WaveFunctions/Ames/(.*)_Ames_wavefunction.npy', file)
    ID = pre_ID[0]
    ID_list.append(int(ID))

calculated_num = len(ID_list)

#Error分子
Error_mols = pd.read_table('Ames_ErrorMolecule.txt', sep = ' ')
Error_ID = Error_mols.ID
ID_list.extend(Error_ID)

#データの編集
drop_list = ID_list
calc_datasets = edited_dataset.drop(drop_list)

#データ数の取得
data_num = len(calc_datasets.index)

print('start with {} samples'.format(data_num))

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
while len(calc_datasets.index) > 0: 
    #forループ
    #コピーの作成
    loop_datasets = calc_datasets.copy()
    for i, (ID, smile) in enumerate(zip(loop_datasets['ID'], loop_datasets['smiles'])):
        try:
            #logfile
            print('ID: {}, smiles: {}'.format(ID, smile))
            try:
                psi4.set_output_file("./log_files/{}.log".format(ID))
            except SystemError:
                print("rise system Error in line 79")
                #errorした分子の取得と削除
                calc_datasets.drop(ID-1, inplace = True)
                rm_num += 1
                task_num = data_num - rm_num - calc_num
                print('Remaining molecules:', task_num)
                psi4.core.opt_clean()
                print('skiped this molecule: ', smile)
                print('Error passed')
                break 
            #smilesから三次元構造を作る→雑に構造最適化
            mol = Chem.MolFromSmiles(smile)   #Molオブジェクトの生成
            mol = Chem.AddHs(mol)   #水素付加
            params = ETKDGv3()
            params.randomseed = 1
            EmbedMolecule(mol, params)   #三次元に拡張
            #MMFF（Merck Molecular Force Fieldで構造最適化
            MMFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
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
                print("raise OptimizationConvergenceError in line 120")
                #errorした分子の取得と削除
                calc_datasets.drop(ID-1, inplace = True)
                rm_num += 1
                task_num = data_num - rm_num - calc_num
                print('Remaining molecules:', task_num)
                psi4.core.opt_clean()
                print('skiped this molecule: ', smile)
                print('Error passed')
                break
            except TimeoutError:
                print("raise TimeoutError in line 120")
                #errorした分子の取得と削除
                calc_datasets.drop(ID-1, inplace = True)
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
            calc_datasets.drop(ID-1, inplace = True)
            rm_num += 1
            task_num = data_num - rm_num - calc_num
            print('Remaining molecules:', task_num)
            psi4.core.opt_clean()
            print('skiped this molecule: ', smile)
            print('Error passed')
            break
        except Exception:
            print('Rise Other Error')
            calc_datasets.drop(ID-1, inplace = True)
            rm_num += 1
            task_num = data_num - rm_num - calc_num
            print('Remaining molecules:', task_num)
            psi4.core.opt_clean()
            print('skiped this molecule: ', smile)
            print('Error passed')
            break
            
        #wavefunctionの書き出し
        wave_function.to_file("../WaveFunctions/Ames/{}_Ames_wavefunction".format(ID))
        #計算の終わった分子の削除
        calc_datasets.drop(ID-1, inplace = True)
        calc_num += 1
        task_num = data_num - rm_num - calc_num
        print("completed caluculations: ", calc_num)
        print('Remaining molecules:', task_num)
        
print("Calculation was finished!!")