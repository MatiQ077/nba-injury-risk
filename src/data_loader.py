import pandas as pd
from .config import DATA_FILE

def load_dataset(path=DATA_FILE) -> pd.DataFrame:    
    df = pd.read_csv(path)
    return df