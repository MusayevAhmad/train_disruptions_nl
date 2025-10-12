import pandas as pd
from pathlib import Path
from logger import Logger


class DataLoader:
    def __init__(self, datasets_path, logger=None):
        self.datasets_path = Path(datasets_path)
        self.logger = logger or Logger()

    def load_data(self):
        self.logger.section("LOADING DATA")
        csv_files = sorted(self.datasets_path.glob('disruptions-20*.csv'))
        if not csv_files:
            self.logger.info("⚠️  No disruptions-20*.csv files found. Falling back to disruptions-2024.csv")
            csv_files = [self.datasets_path / 'disruptions-2024.csv']

        self.logger.files_found(csv_files)
        frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            frames.append(df)
            self.logger.file_loaded(csv_file.name, len(df))

        combined = pd.concat(frames, ignore_index=True)
        self.logger.combined_dataset(len(combined), len(combined.columns))
        return combined


