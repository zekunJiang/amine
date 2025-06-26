"""
胺分子数据集生成器
生成用于CO2吸收反应能垒预测的胺分子SMILES数据集

包含:
- 伯胺 (Primary amines): R-NH2
- 仲胺 (Secondary amines): R-NH-R'  
- 叔胺 (Tertiary amines): R-N(R'-R'')
- 各种功能化胺类化合物
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

class AmineDatasetGenerator:
    """胺分子数据集生成器"""
    
    def __init__(self):
        self.primary_amines = []
        self.secondary_amines = []
        self.tertiary_amines = []
        
    def generate_primary_amines(self):
        """生成伯胺化合物"""
        # 常见的伯胺
        primary_amines = [
            # 简单脂肪族伯胺
            'CN',  # 甲胺
            'CCN',  # 乙胺
            'CCCN',  # 丙胺
            'CCCCN',  # 丁胺
            'CCCCCN',  # 戊胺
            'CCCCCCN',  # 己胺
            'CCCCCCCN',  # 庚胺
            'CCCCCCCCN',  # 辛胺
            
            # 支链伯胺
            'CC(C)N',  # 异丙胺
            'CCC(C)N',  # 2-丁胺
            'CC(C)CN',  # 异丁胺
            'CC(C)(C)CN',  # 新戊胺
            
            # 环状伯胺
            'C1CCNCC1',  # 哌嗪-4-胺
            'NC1CCCC1',  # 环戊胺
            'NC1CCCCC1',  # 环己胺
            
            # 功能化伯胺
            'NCCCO',  # 3-氨基-1-丙醇
            'NCCO',  # 乙醇胺 (MEA)
            'NCCN',  # 乙二胺 (EDA)
            'NCCCN',  # 1,3-丙二胺
            'NCCCCN',  # 1,4-丁二胺
            'NCCCCCN',  # 1,5-戊二胺
            'NCCCCCCN',  # 1,6-己二胺
            
            # 氨基酸相关
            'NCC(=O)O',  # 甘氨酸
            'NC(C)C(=O)O',  # 丙氨酸
            'NC(CO)C(=O)O',  # 丝氨酸
            
            # 芳香族伯胺
            'Nc1ccccc1',  # 苯胺
            'Nc1ccc(C)cc1',  # 对甲苯胺
            'Nc1ccc(O)cc1',  # 对氨基苯酚
            'Nc1ccc(N)cc1',  # 对苯二胺
            
            # 杂环胺
            'Nc1ccccn1',  # 2-氨基吡啶
            'Nc1cccnc1',  # 3-氨基吡啶
            'Nc1ccncc1',  # 4-氨基吡啶
        ]
        
        return primary_amines
    
    def generate_secondary_amines(self):
        """生成仲胺化合物"""
        secondary_amines = [
            # 简单脂肪族仲胺
            'CNC',  # 二甲胺
            'CCNCC',  # 二乙胺
            'CCCNCCC',  # 二丙胺
            'CCCCNCCCC',  # 二丁胺
            
            # 不对称仲胺
            'CNCC',  # N-甲基乙胺
            'CNCCC',  # N-甲基丙胺
            'CCNCCC',  # N-乙基丙胺
            'CNCCCC',  # N-甲基丁胺
            
            # 环状仲胺
            'C1CCNCC1',  # 哌啶
            'C1CNCCC1',  # 吡咯烷
            'C1CNCCCC1',  # 六氢氮杂卓
            
            # 功能化仲胺
            'CNCCCO',  # N-甲基-3-氨基-1-丙醇
            'CCNCCCO',  # N-乙基-3-氨基-1-丙醇
            'OCCNCCO',  # 二乙醇胺 (DEA)
            'CNCCN',  # N-甲基乙二胺
            'CCNCCN',  # N-乙基乙二胺
            
            # 哌嗪衍生物
            'CN1CCNCC1',  # N-甲基哌嗪
            'CCN1CCNCC1',  # N-乙基哌嗪
            'C1CN(CCO)CCN1',  # N-(2-羟乙基)哌嗪
            
            # 芳香族仲胺
            'CNc1ccccc1',  # N-甲基苯胺
            'CCNc1ccccc1',  # N-乙基苯胺
            'c1ccc(Nc2ccccc2)cc1',  # 二苯胺
            
            # 杂环仲胺
            'CN1CCCC1',  # N-甲基吡咯烷
            'CCN1CCCC1',  # N-乙基吡咯烷
            'CN1CCCCC1',  # N-甲基哌啶
        ]
        
        return secondary_amines
    
    def generate_tertiary_amines(self):
        """生成叔胺化合物"""
        tertiary_amines = [
            # 简单脂肪族叔胺
            'CN(C)C',  # 三甲胺
            'CCN(CC)CC',  # 三乙胺
            'CCCN(CCC)CCC',  # 三丙胺
            'CCCCN(CCCC)CCCC',  # 三丁胺
            
            # 不对称叔胺
            'CN(C)CC',  # N,N-二甲基乙胺
            'CCN(C)C',  # N,N-二甲基乙胺
            'CN(CC)CCC',  # N-甲基-N-乙基丙胺
            'CCN(CCC)CCCC',  # N-乙基-N-丙基丁胺
            
            # 功能化叔胺
            'CN(C)CCO',  # N,N-二甲基乙醇胺
            'CCN(CC)CCO',  # N,N-二乙基乙醇胺
            'OCCN(CCO)CCO',  # 三乙醇胺 (TEA)
            'CN(C)CCCO',  # 3-(二甲基氨基)-1-丙醇
            'CCN(CC)CCCO',  # 3-(二乙基氨基)-1-丙醇
            
            # 环状叔胺
            'CN1CCCCC1',  # N-甲基哌啶
            'CCN1CCCCC1',  # N-乙基哌啶
            'CN1CCCC1',  # N-甲基吡咯烷
            'CCN1CCCC1',  # N-乙基吡咯烷
            
            # 双环和多环叔胺
            'CN1CCN(C)CC1',  # N,N'-二甲基哌嗪
            'CCN1CCN(CC)CC1',  # N,N'-二乙基哌嗪
            'C1CN2CCN1CC2',  # 1,4-二氮杂双环[2.2.2]辛烷 (DABCO)
            
            # 芳香族叔胺
            'CN(C)c1ccccc1',  # N,N-二甲基苯胺
            'CCN(CC)c1ccccc1',  # N,N-二乙基苯胺
            'CN(C)c1ccc(C)cc1',  # N,N-二甲基-4-甲基苯胺
            
            # 季铵化合物前体
            'C[N+](C)(C)CCO',  # 胆碱型化合物
            'CCN(CC)CC[N+](C)(C)C',  # 双功能叔胺
            
            # 杂环叔胺
            'Cn1ccnc1',  # N-甲基咪唑
            'CCn1ccnc1',  # N-乙基咪唑
            'CN1C=CN=C1',  # N-甲基咪唑（异构体）
        ]
        
        return tertiary_amines
    
    def generate_specialized_co2_capture_amines(self):
        """生成专门用于CO2捕获的胺类化合物"""
        specialized_amines = [
            # 常用CO2吸收剂
            'NCCO',  # 单乙醇胺 (MEA)
            'OCCNCCO',  # 二乙醇胺 (DEA)
            'OCCN(CCO)CCO',  # 三乙醇胺 (TEA)
            'CC(O)CN',  # 1-氨基-2-丙醇 (MIPA)
            'NCCCCO',  # 4-氨基-1-丁醇 (AMP)
            'CN(CCO)CCO',  # N-甲基二乙醇胺 (MDEA)
            
            # 立体阻碍胺
            'CC(C)(N)CCO',  # 2-氨基-2-甲基-1-丙醇 (AMP)
            'CC(C)(CN)C(C)(C)N',  # 2,2'-二氨基二异丙胺
            
            # 环状胺CO2吸收剂
            'C1CCN(CCO)CC1',  # 4-(2-羟乙基)哌啶
            'C1CN(CCO)CCN1',  # 1-(2-羟乙基)哌嗪
            'OCC1CCCCN1',  # 2-哌啶甲醇
            
            # 双功能胺
            'NCCCN',  # 1,3-丙二胺
            'NCCCCN',  # 1,4-丁二胺
            'NCCOCCN',  # 双(2-氨乙基)醚
            'NCCOCCOCCOCN',  # 聚乙二醇双胺
            
            # 氨基酸盐相关
            'NCC(=O)[O-]',  # 甘氨酸阴离子
            'NC(C)C(=O)[O-]',  # 丙氨酸阴离子
            'NC(CCC(=O)[O-])C(=O)[O-]',  # 谷氨酸阴离子
            
            # 咪唑基化合物
            'c1c[nH]cn1',  # 咪唑
            'Cc1c[nH]cn1',  # 4-甲基咪唑
            'c1cn(CCO)cn1',  # 1-(2-羟乙基)咪唑
            
            # 胍基化合物
            'NC(=N)N',  # 胍
            'CNC(=N)N',  # 1-甲基胍
            'NC(=N)NCC(=O)O',  # 胍基乙酸
        ]
        
        return specialized_amines
    
    def validate_smiles(self, smiles_list):
        """验证SMILES有效性"""
        valid_smiles = []
        invalid_count = 0
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 计算一些基本属性确保分子合理
                    mw = Descriptors.MolWt(mol)
                    if 10 < mw < 500:  # 分子量在合理范围内
                        valid_smiles.append(smiles)
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
                
        print(f"验证完成: {len(valid_smiles)} 个有效SMILES, {invalid_count} 个无效")
        return valid_smiles
    
    def classify_amine_type(self, smiles):
        """分类胺的类型"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 'unknown'
            
            # 查找氮原子
            nitrogen_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
            
            if not nitrogen_atoms:
                return 'no_nitrogen'
            
            # 分析氮原子的连接情况
            for n_atom in nitrogen_atoms:
                # 获取与氮相连的非氢原子数
                heavy_neighbors = len([neighbor for neighbor in n_atom.GetNeighbors() 
                                     if neighbor.GetAtomicNum() != 1])
                
                # 伯胺：氮连接1个非氢原子
                if heavy_neighbors == 1:
                    return 'primary'
                # 仲胺：氮连接2个非氢原子
                elif heavy_neighbors == 2:
                    return 'secondary'
                # 叔胺：氮连接3个非氢原子
                elif heavy_neighbors == 3:
                    return 'tertiary'
                    
            return 'other'
            
        except:
            return 'unknown'
    
    def generate_complete_dataset(self):
        """生成完整的胺分子数据集"""
        print("正在生成胺分子数据集...")
        
        # 收集所有胺类化合物
        all_amines = []
        
        # 生成各类胺
        primary = self.generate_primary_amines()
        secondary = self.generate_secondary_amines()
        tertiary = self.generate_tertiary_amines()
        specialized = self.generate_specialized_co2_capture_amines()
        
        all_amines.extend(primary)
        all_amines.extend(secondary)
        all_amines.extend(tertiary)
        all_amines.extend(specialized)
        
        # 去重
        unique_amines = list(set(all_amines))
        print(f"去重前: {len(all_amines)} 个分子")
        print(f"去重后: {len(unique_amines)} 个分子")
        
        # 验证SMILES
        valid_amines = self.validate_smiles(unique_amines)
        
        # 创建DataFrame
        data = []
        for i, smiles in enumerate(valid_amines):
            amine_type = self.classify_amine_type(smiles)
            data.append({
                'molecule_id': f'amine_{i+1:03d}',
                'smiles': smiles,
                'amine_type': amine_type,
                'description': self._get_molecule_description(smiles)
            })
            
        df = pd.DataFrame(data)
        
        # 检查类别分布并补充样本
        type_counts = df['amine_type'].value_counts()
        print("\n初始胺类型分布:")
        for amine_type, count in type_counts.items():
            print(f"  {amine_type}: {count} 个")
        
        # 为样本数过少的类别补充样本
        min_samples_needed = 5  # 每个类别至少5个样本
        supplemented_data = []
        
        for amine_type in type_counts.index:
            if type_counts[amine_type] < min_samples_needed:
                print(f"\n⚠️  {amine_type} 类别样本不足（{type_counts[amine_type]}个），正在补充...")
                
                # 基于现有分子生成变体
                existing_molecules = df[df['amine_type'] == amine_type]['smiles'].tolist()
                needed = min_samples_needed - type_counts[amine_type]
                
                # 简单的分子变体生成（通过添加甲基等小的修饰）
                variations = self._generate_molecular_variations(existing_molecules, needed)
                
                for j, var_smiles in enumerate(variations):
                    if self.validate_smiles([var_smiles]):  # 验证变体有效性
                        supplemented_data.append({
                            'molecule_id': f'{amine_type}_var_{j+1:02d}',
                            'smiles': var_smiles,
                            'amine_type': amine_type,
                            'description': f'{amine_type}变体分子'
                        })
        
        # 合并补充的数据
        if supplemented_data:
            supp_df = pd.DataFrame(supplemented_data)
            df = pd.concat([df, supp_df], ignore_index=True)
            print(f"补充了 {len(supplemented_data)} 个变体分子")
        
        # 最终统计信息
        final_counts = df['amine_type'].value_counts()
        print("\n最终胺类型分布:")
        for amine_type, count in final_counts.items():
            print(f"  {amine_type}: {count} 个")
        print(f"\n总分子数: {len(df)}")
        
        return df
    
    def _generate_molecular_variations(self, base_molecules, needed_count):
        """生成分子变体"""
        variations = []
        modification_patterns = [
            ('C', 'CC'),  # 甲基化
            ('N', 'N(C)'),  # N-甲基化
            ('O', 'OC'),  # 乙基化羟基
        ]
        
        for base_smiles in base_molecules:
            if len(variations) >= needed_count:
                break
                
            for old, new in modification_patterns:
                if len(variations) >= needed_count:
                    break
                    
                # 简单的字符串替换（实际应用中可能需要更复杂的化学修饰）
                if old in base_smiles and base_smiles.count(old) == 1:
                    modified = base_smiles.replace(old, new, 1)
                    if modified != base_smiles and modified not in base_molecules:
                        variations.append(modified)
        
        return variations[:needed_count]
    
    def _get_molecule_description(self, smiles):
        """获取分子描述"""
        descriptions = {
            'CN': '甲胺',
            'CCN': '乙胺',
            'NCCO': '单乙醇胺(MEA)',
            'OCCNCCO': '二乙醇胺(DEA)',
            'OCCN(CCO)CCO': '三乙醇胺(TEA)',
            'CN(CCO)CCO': 'N-甲基二乙醇胺(MDEA)',
            'CN(C)C': '三甲胺',
            'CCN(CC)CC': '三乙胺',
            'c1c[nH]cn1': '咪唑',
            'NCCCN': '1,3-丙二胺',
            'NC1CCCCC1': '环己胺'
        }
        
        return descriptions.get(smiles, f'胺类化合物({smiles})')


def main():
    """主函数"""
    print("="*60)
    print("胺分子数据集生成器")
    print("用于CO2吸收反应能垒预测")
    print("="*60)
    
    # 创建生成器
    generator = AmineDatasetGenerator()
    
    # 生成数据集
    dataset = generator.generate_complete_dataset()
    
    # 保存为CSV
    output_file = 'input_molecules.csv'
    dataset.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ 数据集已保存为: {output_file}")
    print(f"📊 包含 {len(dataset)} 个胺分子")
    
    # 显示样例数据
    print("\n📋 数据样例:")
    print(dataset.head(10).to_string(index=False))
    
    # 另外生成CO2分子数据
    co2_data = pd.DataFrame({
        'molecule_id': ['CO2_001'],
        'smiles': ['O=C=O'],
        'amine_type': ['reactant'],
        'description': ['二氧化碳']
    })
    
    # 合并数据集包含CO2
    complete_dataset = pd.concat([dataset, co2_data], ignore_index=True)
    complete_output = 'input_molecules_with_co2.csv'
    complete_dataset.to_csv(complete_output, index=False, encoding='utf-8')
    
    print(f"\n✅ 包含CO2的完整数据集已保存为: {complete_output}")
    print("\n🎯 数据集已准备完成，可用于后续的反应能垒预测!")


if __name__ == "__main__":
    main() 