import os
import numpy as np
from Bio.PDB import PDBParser
import torch
from torch.utils.data import Dataset

class MDTrajectoryDataset(Dataset):
    def __init__(self, result_directory):
        self.pdb_files = self.load_pdb_files_from_directory(result_directory)
        self.max_atoms = self.get_max_atoms()  # 최대 원자 수를 계산합니다.

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_files = self.pdb_files[idx]
        frames = []
        torsion_angles = []

        for pdb_file in pdb_files:
            atoms = self.extract_coordinates_from_pdb(pdb_file)
            frames.append(atoms)

            # angles = self.calculate_torsion_angles(pdb_file)
            # torsion_angles.append(angles)

        # 각 프레임에서 원자 좌표를 (N, 3)로 만들어 (S, N, 3) 형태로 변환
        padded_frames = self.pad_frames(frames)
        # padded_torsion_angles = self.pad_torsion_angles(torsion_angles)

        # torsion = torch.tensor(padded_torsion_angles, dtype=torch.float32) 
        
        translation = torch.tensor(padded_frames, dtype=torch.float32) # (S, N ,3)   S : Frame    N : Number of the atom   3 : x, y, z

        return translation, pdb_files  #, torsion

    @staticmethod
    def load_pdb_files_from_directory(directory):

        molecule_dict = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdb'):
                    molecule_name = os.path.basename(file).split('_')[0]  # 분자 이름 추출
                    if molecule_name not in molecule_dict:
                        molecule_dict[molecule_name] = []
                    molecule_dict[molecule_name].append(os.path.join(root, file))
        return list(molecule_dict.values())

    @staticmethod
    def extract_coordinates_from_pdb(pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        atom_coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_coords.append(atom.coord)

        return np.array(atom_coords)

    def get_max_atoms(self):
        max_atoms = 0
        for pdb_files in self.pdb_files:
            for pdb_file in pdb_files:
                atom_count = len(self.extract_coordinates_from_pdb(pdb_file))
                max_atoms = max(max_atoms, atom_count)
        #max_atoms = 1000 / 나중에 상수로 지정 가능
        return max_atoms

    def pad_frames(self, frames):
        padded = []
        for atoms in frames:
            # 원자 수가 최대 원자 수에 맞춰 패딩
            if len(atoms) < self.max_atoms:
                padding = np.zeros((self.max_atoms - len(atoms), 3))
                padded.append(np.vstack((atoms, padding)))  # 원자 좌표와 패딩을 합칩니다.
            else:
                padded.append(atoms)
        
        # (S, N, 3) 형태로 변환
        return np.array(padded)
    
    @staticmethod
    def calculate_torsion_angles(self, pdb_file):
        # PDB 파일에서 원자 좌표를 추출합니다.
        atoms = self.extract_coordinates_from_pdb(pdb_file)
        
        torsion_angles = []
        
        # 각 원자에 대한 인덱스 (예시: A, U, G, C의 경우)
        # 이 인덱스는 RNA의 특정 원자에 맞게 조정해야 합니다.
        for i in range(1, len(atoms) - 3):
            # 4개의 원자를 선택하여 토션 각을 계산
            a1 = atoms[i - 1]  # 이전 원자
            a2 = atoms[i]      # 현재 원자
            a3 = atoms[i + 1]  # 다음 원자
            a4 = atoms[i + 2]  # 그 다음 원자
            
            # 토션 각 계산
            angle = self.calculate_dihedral(a1, a2, a3, a4)
            torsion_angles.append(angle)

        # 결과를 (N, 7) 형태로 변환
        # 각 프레임에 대해 7개의 각이 필요하므로 패딩 추가
        torsion_angles = np.array(torsion_angles)
        if torsion_angles.shape[0] < 7:
            # 부족한 경우에는 0으로 패딩
            padding = np.zeros((7 - torsion_angles.shape[0], 7))
            torsion_angles = np.vstack((torsion_angles, padding))
        
        return torsion_angles

    @staticmethod
    def calculate_dihedral(self, a1, a2, a3, a4):
        # 4개의 원자 좌표로부터 다이헤드럴 각을 계산합니다.
        b1 = a2 - a1
        b2 = a3 - a2
        b3 = a4 - a3

        # 벡터의 외적을 이용하여 법선 벡터 계산
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # 법선 벡터의 단위 벡터 계산
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        # 각도 계산
        angle = np.arctan2(np.dot(n1, b2) * np.linalg.norm(b3), np.dot(n1, n2))
        return np.degrees(angle)  # 도 단위로 반환
    
    @staticmethod
    def pad_torsion_angles(self, torsion_angles):
        # 토션 각을 패딩하여 (S, N, 7) 형태로 변환하는 로직을 구현합니다.
        pass

if __name__ == '__main__':
    result_directory = './Test_data'
    dataset = MDTrajectoryDataset(result_directory)

    print("Dataset size:", len(dataset))

    # 모든 분자에 대한 프레임 데이터 출력
    for idx in range(len(dataset)):
        frames = dataset[idx][0]  # (S, N, 3) 형태의 텐서
        print(f"Data for molecule {idx + 1}:")
        print("Frames shape:", frames.shape)  # (S, N, 3) 형태 확인
        
        # 각 프레임의 원자 좌표 출력
        for s in range(frames.shape[0]):  # 각 프레임에 대해
            print(f"Frame {s + 1}:")
            # print(frames[s])  # (N, 3) 형태의 원자 좌표 출력
            print(frames[s].shape)
        print("\n")  # 각 분자 데이터 사이에 빈 줄 추가
    
    # def mol2graph(self, mol):
    #     # atoms
    #     atom_features_list = []
    #     for atom in mol.GetAtoms():
    #         atom_features_list.append(atom_to_feature_vector(atom))
    #     x = np.array(atom_features_list, dtype=np.int64)

    #     coords = mol.GetConformer().GetPositions()
    #     z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    #     # bonds
    #     num_bond_features = 3  # bond type, bond stereo, is_conjugated
    #     if len(mol.GetBonds()) > 0:  # mol has bonds
    #         edges_list = []
    #         edge_features_list = []
    #         for bond in mol.GetBonds():
    #             i = bond.GetBeginAtomIdx()
    #             j = bond.GetEndAtomIdx()

    #             edge_feature = bond_to_feature_vector(bond)

    #             # add edges in both directions
    #             edges_list.append((i, j))
    #             edge_features_list.append(edge_feature)
    #             edges_list.append((j, i))
    #             edge_features_list.append(edge_feature)

    #         # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    #         edge_index = np.array(edges_list, dtype=np.int64).T

    #         # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #         edge_attr = np.array(edge_features_list, dtype=np.int64)

    #     else:  # mol has no bonds
    #         edge_index = np.empty((2, 0), dtype=np.int64)
    #         edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    #     graph = dict()
    #     graph["edge_index"] = edge_index
    #     graph["edge_feat"] = edge_attr
    #     graph["node_feat"] = x
    #     graph["num_nodes"] = len(x)
    #     graph["position"] = coords
    #     graph["z"] = z

    #     return graph
