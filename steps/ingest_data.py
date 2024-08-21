import logging

import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df

    except Exception as e:
        logging.error(f"Error while reading the data: {e}")
        raise e