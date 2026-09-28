import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List

class AdvancedTabularDataset(Dataset):
    """
    Pipeline de dados otimizado para lidar com variáveis heterogêneas.
    Separa e converte dados contínuos e categóricos em tensores dinamicamente.
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        continuous_cols: List[str], 
        categorical_cols: List[str], 
        target_col: str
    ):
        # Trabalhamos com arrays NumPy internos por questão de velocidade de indexação
        self.x_cont = dataframe[continuous_cols].values.astype(np.float32)
        self.x_cat = dataframe[categorical_cols].values.astype(np.int64)
        self.labels = dataframe[target_col].values.astype(np.int64)
        
    def __len__(self) -> int:
        # Retorna o total de registros
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        # Acessado dinamicamente pelo DataLoader do PyTorch
        cont_tensor = torch.tensor(self.x_cont[idx])
        cat_tensor = torch.tensor(self.x_cat[idx])
        label_tensor = torch.tensor(self.labels[idx])
        
        return cont_tensor, cat_tensor, label_tensor