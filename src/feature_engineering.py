import pandas as pd
import numpy as np
from logger import Logger


class FeatureEngineer:
    def __init__(self, logger=None):
        self.logger = logger or Logger()
    def engineer(self, df):
        self.logger.section("FEATURE ENGINEERING")
        df = df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True).dt.tz_localize(None)
        df['year'] = df['start_time'].dt.year

        def assign_hour_bin(hour):
            if 0 <= hour <= 5:
                return 'night'
            if 6 <= hour <= 9:
                return 'morning_peak'
            if 10 <= hour <= 15:
                return 'midday'
            if 16 <= hour <= 19:
                return 'evening_peak'
            return 'late'

        df['hour'] = df['start_time'].dt.hour
        df['hour_bin'] = df['hour'].apply(assign_hour_bin)
        df['weekday'] = df['start_time'].dt.dayofweek
        df['month'] = df['start_time'].dt.month

        def count_stations(station_codes):
            if pd.isna(station_codes) or station_codes == '':
                return 0
            return len([s.strip() for s in str(station_codes).split(',') if s.strip()])

        df['station_count'] = df['rdt_station_codes'].apply(count_stations)

        def normalize_cause_group(value):
            text = "" if pd.isna(value) else str(value).strip().lower()
            if text in ("", "nan"):
                return 'unknown'
            mapping = {
                'rolling stock': 'rolling stock',
                'rolling_stock': 'rolling stock',
                'rollingstock': 'rolling stock',
                'infrastructure': 'infrastructure',
                'accidents': 'accidents',
                'accident': 'accidents',
                'external': 'external',
                'engineering work': 'engineering work',
                'engineering works': 'engineering work',
                'engineering': 'engineering work',
                'logistical': 'logistical',
                'logistics': 'logistical',
                'staff': 'staff',
                'staffing': 'staff',
                'weather': 'weather',
                'unknown': 'unknown',
                'other': 'other',
            }
            return mapping.get(text, 'other')

        df['cause_group'] = df['cause_group'].apply(normalize_cause_group)
        df['is_engineering_work'] = (df['cause_group'] == 'engineering work').astype(int)

        self.logger.features_engineered({
            'hour_bin categories': sorted(df['hour_bin'].unique()),
            'weekday range': f"{df['weekday'].min()}-{df['weekday'].max()}",
            'month range': f"{df['month'].min()}-{df['month'].max()}",
            'station_count range': f"{df['station_count'].min()}-{df['station_count'].max()}",
            'cause_group categories': sorted(df['cause_group'].unique()),
            'is_engineering_work': f"{df['is_engineering_work'].sum()} positives",
        })
        return df

    def compute_threshold(self, df):
        self.logger.section("COMPUTING THRESHOLD")
        available_years = sorted(df['year'].unique())
        if all(year in available_years for year in [2021, 2022, 2023, 2024]):
            df_threshold = df[(df['year'].isin([2021, 2022, 2023, 2024])) & (df['duration_minutes'].notna())]
            self.logger.threshold_info([2021, 2022, 2023, 2024], len(df_threshold))
        else:
            df_threshold = df[df['duration_minutes'].notna()]
            self.logger.threshold_info(None, len(df_threshold))

        threshold = df_threshold['duration_minutes'].median()
        self.logger.threshold_result(threshold)
        return threshold

    def create_target(self, df, threshold):
        df = df.copy()
        df['is_long'] = (df['duration_minutes'] >= threshold).astype(int)
        n_long = df['is_long'].sum()
        n_total = len(df)
        self.logger.class_distribution(threshold, n_long, n_total)
        return df

    def split(self, df):
        self.logger.section("TRAIN/VALIDATION/TEST SPLIT")
        available_years = sorted(df['year'].unique())
        has_multi_year = all(year in available_years for year in [2021, 2022, 2023, 2024])

        if has_multi_year:
            train_df = df[df['year'].isin([2021, 2022, 2023])].copy()
            val_df = df[(df['year'] == 2023) & (df['month'].isin([10, 11, 12]))].copy()
            test_df = df[df['year'] == 2024].copy()
            train_df = train_df[~train_df.index.isin(val_df.index)]
            split_scheme = "multi-year (2021-2023 train, 2023 Q4 val, 2024 test)"
        else:
            df_sorted = df.sort_values('start_time').reset_index(drop=True)
            n = len(df_sorted)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
            split_scheme = "2024-only (earliest 80% train, next 10% val, latest 10% test)"

        self.logger.split_info(available_years, split_scheme)
        self.logger.split_sizes(
            len(train_df), len(val_df), len(test_df),
            100 * train_df['is_long'].mean(),
            100 * val_df['is_long'].mean(),
            100 * test_df['is_long'].mean()
        )
        return train_df, val_df, test_df, split_scheme


