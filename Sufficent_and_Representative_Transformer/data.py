import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Iterable, Tuple
from torch.utils.data import Dataset
import torch
import h5py

#Adapted from https://github.com/KatherLab/marugoto/blob/main/marugoto/mil/data.py
def get_cohort_df(
    clini_table: Path,            # Path to the clinical table file (CSV or Excel format).
    slide_csv: Path,              # Path to the slide CSV file.
    feature_dir: Path,            # Path to the directory containing slide feature files.
    target_labels: Iterable[str], # List of target labels.
    label_dict: dict,             # Dict of mappings from label names to numerical targets.
    cohort: str,                  #The cohort name (e.g., 'TCGA'). 
) -> pd.DataFrame:

    clini_df = pd.read_csv(
        clini_table, dtype=str
    ) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')


    # remove columns not in target_labels
    for key in df.columns:
        if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME']:
            df.drop(key, axis=1, inplace=True)
    
    # remove rows/slides with non-valid labels
    for target in target_labels:
        df = df.dropna(subset=target)
        df[target] = df[target].map(lambda p: int(p) if p.isdigit() else label_dict[p])
    
    # remove slides we don't have
    h5s = set(feature_dir.glob('**/*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    
    h5_df['FILENAME'] = h5_df.slide_path.map(
        lambda p: p.stem.split('.')[0].split('_files')[0]+'.'+p.stem.split('.')[1].split('_files')[0]
    )  
    df = df.merge(h5_df, on='FILENAME')
    return df


# Generate a multi-cohort DataFrame concatenating the DataFrame from single cohorts.
def get_multi_cohort_df(
    data_config: Path,                   # Path to the data configuration file.
    cohorts: Iterable[str],              # List of cohorts to include in the multi-cohort DataFrame.
    target_labels: Iterable[str],        # List of target labels.
    label_dict: dict,                    # Dict of mappings from label names to numerical targets.
    norm: str = 'macenko',               # Normalization method. Defaults to 'macenko'.
    feats: str = 'ctranspath',           # feats (str, optional): Feature extractor used. Defaults to 'ctranspath'.
) -> Tuple[pd.DataFrame, dict]:

    df_list = []
    np_list = []

    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        for cohort in cohorts:
            clini_table = Path(data_config[cohort]['clini_table'])
            slide_csv = Path(data_config[cohort]['slide_csv'])
            feature_dir = Path(data_config[cohort]['feature_dir'][norm][feats])
            current_df = get_cohort_df(
                clini_table, slide_csv, feature_dir, target_labels, label_dict, cohort
            )
            df_list.append(current_df)
            np_list.append(len(current_df.PATIENT))
    data = pd.concat(df_list, ignore_index=True)
    if len(data.PATIENT) != sum(np_list):
        print(
            f'number of patients in joint dataset {len(data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}'
        )
    return data

# Using broadcast and vectorization operations to calculate the relative distance matrix
def euclidean_distance_matrix(coords):
    n = coords.size(0)
    diff = coords[:, 1:, None] - coords[:, 1:, None].transpose(0, 2)
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum(dim=1)
    distances = torch.sqrt(sum_squared_diff)
    return distances

class MILDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,              # The input data containing the labels and paths to feature vectors.
        indices: Iterable[int],          # The indices to select from the input data.
        target_labels: Iterable[str],    # The target labels to be predicted.
        norm: str = 'macenko'            # The normalization method for features. Defaults to 'macenko'.
    ):
        self.data = data.iloc[indices]
        self.indices = indices
        self.target_labels = target_labels
        self.norm = norm
        
    # update the feature from the feature_index form argumentation
    def __getitem__(self, item):
        
        origin_dir ="/work1/lhj/project/InfoTransformer/hrd_for_3_quantile_8K/2_kmeans_clustering_CE/30n_1200s/features/breast_20X_breast/h5_files/256px_dino_0.5mpp_0xdown_normal"
        # load features and coords from .h5 file
        h5_path = self.data.slide_path[self.indices[item]]

        h5_file = h5py.File(h5_path)

        try:
            feature_index=h5_file['feats_index'][:]
        except TypeError or KeyError:
            print(h5_path+"has no feats_index")
        
        name=str(h5_path).split("/")[-1].split("_")[0]
        h5_file_path=origin_dir+"/"+name+".h5"

        with h5py.File(h5_file_path,"r") as f:
            features=torch.Tensor(np.array(f['feats'])[feature_index])
            coords = torch.Tensor(np.array(f['coords'])[feature_index])
        tiles = torch.tensor([features.shape[0]]) 
        
        # create binary or numeric labels from categorical labels
        label = [self.data[target][self.indices[item]] for target in self.target_labels]
        label = torch.Tensor(label).long().squeeze()

        patient = self.data.PATIENT[self.indices[item]]
        original_coords=coords
        coords[:, 1:] = coords[:, 1:] / coords[:, 1:].max(dim=0).values
        distances_matrix = euclidean_distance_matrix(coords)
        if distances_matrix is None:
            return None
        distances_matrix = torch.Tensor(np.array(distances_matrix))
        return features, original_coords, label, tiles, patient, distances_matrix

    def __len__(self):
        return len(self.data)
