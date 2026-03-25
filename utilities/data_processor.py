import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
import warnings

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Class for handling data preprocessing for tasks 1.3 and 2.3b
    Handles unit fixing, outlier removal, missing value imputation, and feature scaling.
    
    This transformer is designed to be fully compatible with Scikit-Learn pipelines.
    It memorizes statistics (like medians and scale factors) during `fit` and applies 
    them strictly during `transform` to prevent target leakage.
    
    Parameters
    ----------
    impute_outliers : bool, default=True
        If True, evaluates physiological plausibility of out-of-bounds values. 
        Implausible outliers (sensor blips) are removed and imputed.
    impute_missing : bool, default=False
        If True, performs forward-filling on dynamic variables and median 
        fallback imputation on remaining missing values.
        
    Attributes
    ----------
    first_val_medians_ : dict
        Medians of the first recorded timestamps, grouped by Gender and AgeBin.
    global_medians_ : dict
        Global cohort medians across all timestamps, grouped by Gender and AgeBin.
    agebin_medians_ : dict
        Global cohort medians grouped solely by AgeBin (used as a fallback).
    overall_medians_ : dict
        Global column medians across the entire dataset (absolute fallback).
    highly_skewed_ : list
        List of numerical variables identified as highly skewed (skew < -1 or > 1).
    normal_vars_ : list
        List of numerical variables with normal skewness.
    power_transformer_ : PowerTransformer
        Fitted Yeo-Johnson transformer for highly skewed variables.
    robust_scaler_ : RobustScaler
        Fitted RobustScaler applied after the PowerTransformer.
    standard_scaler_ : StandardScaler
        Fitted StandardScaler for normal variables.
    dummy_columns_ : list
        The list of columns generated after calling pd.get_dummies during fit. 
        Used to enforce aligned column structures during transform.
    """
    def __init__(self, impute_outliers=True, impute_missing=False):
        self.impute_outliers = impute_outliers
        self.impute_missing = impute_missing
        
        self.physiologically_plausible = {
            'Albumin': [0.1, 10.0],       'ALP': [0, 10000],           'ALT': [0, 20000],
            'AST': [0, 20000],           'Bilirubin': [0, 150],       'BUN': [0.5, 300],
            'Cholesterol': [10, 1000],   'Creatinine': [0.05, 40],    'DiasABP': [1, 300],
            'FiO2': [0.21, 1.0],        'GCS': [3, 15],              'Glucose': [5, 2000],
            'HCO3': [2, 100],            'HCT': [5, 90],              'HR': [0, 250],
            'K': [0.5, 15.0],            'Lactate': [0.1, 50],        'Mg': [0.1, 20.0],
            'MAP': [1, 300],             'Na': [80, 210],             'PaCO2': [5, 250],
            'PaO2': [5, 800],            'pH':[6.3, 8.2],             'Platelets': [1, 3000],      
            'RespRate': [1, 150],        'SaO2': [0, 100],           'SysABP': [1, 400],         
            'Temp': [30, 44],            'Urine': [0, 10000],         'WBC': [0, 500]
        }
        
        self.static_vars = ['Age', 'Gender', 'Height', 'Weight']
        self.exclude_vars = ['PatientID', 'Timestamp', 'Label', 'AgeBin']
        self.cat_vars = ['Gender']
        
        self.age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        self.age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109', '110-119']

    def _apply_unit_fixes(self, df):
        """
        Auxiliary method to logically fix physical units of measurement.
        For example: Height mistakenly recorded in meters instead of cm, 
        or pH missing a decimal point. Values extremely out-of-bounds 
        (like Height < -1 or > 50 in meters) are replaced with NaN.
        """
        df_fixed = df.copy()
        
        if 'Height' in df_fixed.columns:
            height_mask_1 = df_fixed['Height'].between(1.3, 2.5)
            df_fixed.loc[height_mask_1, 'Height'] = df_fixed.loc[height_mask_1, 'Height'] * 100
            
            height_mask_2 = df_fixed['Height'].between(10, 25)
            df_fixed.loc[height_mask_2, 'Height'] = df_fixed.loc[height_mask_2, 'Height'] * 10
            
            df_fixed.loc[(df_fixed['Height'] >= -1) & (df_fixed['Height'] <= 50), 'Height'] = np.nan
        
        if 'Weight' in df_fixed.columns:
            df_fixed.loc[(df_fixed['Weight'] >= -1) & (df_fixed['Weight'] <= 15), 'Weight'] = np.nan

        if 'pH' in df_fixed.columns:
            ph_mask = df_fixed['pH'].between(630, 820)
            df_fixed.loc[ph_mask, 'pH'] = df_fixed.loc[ph_mask, 'pH'] / 100
            
        if 'Age' in df_fixed.columns:
            df_fixed['AgeBin'] = pd.cut(df_fixed['Age'], bins=self.age_bins, labels=self.age_labels, right=False)
            
        return df_fixed

    def fit(self, X, y=None):
        """
        Calculates cohort medians, identifies skewed variables, and fits scalers 
        based strictly on the input training dataset.
        
        Process:
        1. Applies unit fixes and isolates physiological variables.
        2. Masks implausible values to NaN to calculate clean baseline medians.
        3. Computes four levels of medians (first-ts, global, agebin, overall) for fallback.
        4. Temporarily imputes data to safely calculate skewness.
        5. Fits PowerTransformer + RobustScaler for skewed variables, and StandardScaler for normal ones.
        6. Extracts generated dummy columns to enforce structure during transform.
        """
        df_clean = X.copy()
        df_clean = self._apply_unit_fixes(df_clean)
        
        self.all_vars = df_clean.columns.tolist()
        self.dynamic_vars = [v for v in self.all_vars if v not in self.static_vars and v not in self.exclude_vars]
        self.num_vars = [v for v in self.all_vars if v not in self.cat_vars and v not in self.exclude_vars]

        for var, (min_val, max_val) in self.physiologically_plausible.items():
            if var in df_clean.columns:
                df_clean.loc[(df_clean[var] < min_val) | (df_clean[var] > max_val), var] = np.nan

        first_vals = df_clean.sort_values(by=['PatientID', 'Timestamp']).groupby('PatientID').first()
        if 'AgeBin' not in first_vals.columns and 'Age' in first_vals.columns:
            first_vals['AgeBin'] = pd.cut(first_vals['Age'], bins=self.age_bins, labels=self.age_labels, right=False)
            
        # Compute different levels of medians for imputation fallback
        self.first_val_medians_ = first_vals.groupby(['Gender', 'AgeBin'], observed=True).median().to_dict('index')
        self.global_medians_ = df_clean.groupby(['Gender', 'AgeBin'], observed=True).median().to_dict('index')
        self.agebin_medians_ = df_clean.groupby('AgeBin', observed=True).median().to_dict('index')
        self.overall_medians_ = df_clean[[v for v in self.num_vars if v in df_clean.columns]].median().to_dict()

        # Perform temporary imputation to safely calculate skewness and scalers without NaNs breaking it
        dfTemp = self._impute_dataframe(df_clean)
        
        # Calculate skewness
        present_num = [v for v in self.num_vars if v in dfTemp.columns]
        if present_num:
            skewness_values = dfTemp[present_num].skew()
            self.highly_skewed_ = skewness_values[(skewness_values > 1) | (skewness_values < -1)].index.tolist()
            self.normal_vars_ = [v for v in present_num if v not in self.highly_skewed_]

            # Fit Scalers
            self.power_transformer_ = PowerTransformer(method='yeo-johnson') if self.highly_skewed_ else None
            self.robust_scaler_ = RobustScaler() if self.highly_skewed_ else None
            self.standard_scaler_ = StandardScaler() if self.normal_vars_ else None

            if self.highly_skewed_:
                pt_out = self.power_transformer_.fit_transform(dfTemp[self.highly_skewed_])
                self.robust_scaler_.fit(pt_out)
            
            if self.normal_vars_:
                self.standard_scaler_.fit(dfTemp[self.normal_vars_])
                
        # Fit dummy columns reference to ensure output matching dummy structures
        present_cat = [v for v in self.cat_vars if v in dfTemp.columns]
        if present_cat:
            dummy_df = pd.get_dummies(dfTemp, columns=present_cat, drop_first=True)
            self.dummy_columns_ = dummy_df.columns.tolist()
        else:
            self.dummy_columns_ = dfTemp.columns.tolist()
            
        return self

    def _resolve_median_fallback(self, gender, agebin, var):
        """
        Auxiliary method to resolve a missing value using a cascading median fallback.
        
        Priority:
        1. Median of the variable at hour 0, matched by Gender and AgeBin.
        2. Median of the variable across all hours, matched by Gender and AgeBin.
        3. Median of the variable across all hours, matched only by AgeBin.
        4. Absolute median of the variable across the entire training dataset.
        """
        val = np.nan
        # 1. Try first_val_medians_ by Gender and AgeBin
        if pd.notna(gender) and pd.notna(agebin):
            key = (gender, agebin)
            if hasattr(self, 'first_val_medians_') and key in self.first_val_medians_ and var in self.first_val_medians_[key]:
                val = self.first_val_medians_[key][var]
                
        # 2. Try global_medians_ by Gender and AgeBin
        if pd.isna(val) and pd.notna(gender) and pd.notna(agebin):
            key = (gender, agebin)
            if hasattr(self, 'global_medians_') and key in self.global_medians_ and var in self.global_medians_[key]:
                val = self.global_medians_[key][var]
                
        # 3. Try agebin_medians_ (if missing gender or above failed)
        if pd.isna(val) and pd.notna(agebin):
            if hasattr(self, 'agebin_medians_') and agebin in self.agebin_medians_ and var in self.agebin_medians_[agebin]:
                val = self.agebin_medians_[agebin][var]
                
        # 4. Overall global median
        if pd.isna(val) and hasattr(self, 'overall_medians_') and var in self.overall_medians_:
            val = self.overall_medians_[var]
            
        return val

    def _impute_dataframe(self, df):
        """
        Auxiliary method to handle base missing value imputation.
        
        1. Forward-fills dynamic variables chronologically per patient.
        2. Imputes any remaining NaNs using the `_resolve_median_fallback` sequence.
        """
        df_processed = df.copy()
        
        valid_dynamic = [v for v in self.dynamic_vars if v in df_processed.columns]
        if valid_dynamic:
            df_processed[valid_dynamic] = df_processed.groupby('PatientID')[valid_dynamic].ffill()

        num_present = [v for v in self.num_vars if v in df_processed.columns]
        if num_present:
            for var in num_present:
                if df_processed[var].isna().any():
                    # Apply the cascading median fallback to any remaining NaNs
                    missing_mask = df_processed[var].isna()
                    for idx in df_processed[missing_mask].index:
                        gender = df_processed.at[idx, 'Gender'] if 'Gender' in df_processed.columns else np.nan
                        agebin = df_processed.at[idx, 'AgeBin'] if 'AgeBin' in df_processed.columns else np.nan
                        
                        df_processed.at[idx, var] = self._resolve_median_fallback(gender, agebin, var)
                        
        return df_processed

    def transform(self, X):
        """
        Applies mathematical normalizations and data cleaning to output a scaled dataframe.
        
        Outlier Imputation Logic:
        - If an out-of-bound measurement returns to normal bounds on the very next reading, 
          it is considered an implausible sensor "blip" and is overwritten.
        - If the outlier persists or lacks future readings, and the patient survived, it is 
          also considered an anomaly and removed. (If Label is unavailable during inference, 
          survival is safely assumed to decouple preprocessing from target leakage).
        - Replaced outliers are overwritten via forward-fill of previously known normal values, 
          or via cascading cohort medians if the outlier occurred on the first timestamp.
          
        Missing Value Imputation Logic:
        - Dynamic features are forward-filled chronologically within a patient's timeline.
        - Any values still missing are filled via the 4-step cascading dictionary lookup:
          `first_val_medians_` -> `global_medians_` -> `agebin_medians_` -> `overall_medians_`.
        
        Scaling Logic:
        - Applies the highly-skewed and normally-skewed scaling pathways strictly using the 
          equations fit during training, guaranteeing zero dataleak from the test distribution.
        - Enforces perfect mock-up alignments for one-hot encoded categories across train/test sets.
        """
        df_processed = X.copy()
        df_processed = self._apply_unit_fixes(df_processed)
        df_processed = df_processed.sort_values(by=['PatientID', 'Timestamp'])

        if self.impute_outliers:
            for var, (min_val, max_val) in self.physiologically_plausible.items():
                if var not in df_processed.columns: continue
                
                is_valid = df_processed[var].between(min_val, max_val) & df_processed[var].notna()
                valid_series = df_processed[var].where(is_valid)
                ffill_values = valid_series.groupby(df_processed['PatientID']).ffill()
                next_valid = valid_series.groupby(df_processed['PatientID']).bfill().shift(-1)
                
                is_outlier = (df_processed[var] < min_val) | (df_processed[var] > max_val)
                
                if 'Label' in df_processed.columns:
                    patient_survived = (df_processed['Label'] == 0)
                else:
                    patient_survived = pd.Series(True, index=df_processed.index)
                    warnings.warn("No 'Label' found on Inference. Assuming survival for outlier plausibility evaluation to prevent target leakage.")
                
                implausible_blip = is_outlier & next_valid.between(min_val, max_val)
                implausible_survived = is_outlier & (~next_valid.between(min_val, max_val) | next_valid.isna()) & patient_survived
                implausible_mask = implausible_blip | implausible_survived
                
                # Decoupled direct overwrite using previously known clean states
                df_processed.loc[implausible_mask, var] = ffill_values.loc[implausible_mask]
                
                # If outlier was first event, ffill exposes NaN. Impute safely via cascading fallback.
                missing_mask = implausible_mask & df_processed[var].isna()
                if missing_mask.any():
                    for idx in df_processed[missing_mask].index:
                        gender = df_processed.at[idx, 'Gender'] if 'Gender' in df_processed.columns else np.nan
                        agebin = df_processed.at[idx, 'AgeBin'] if 'AgeBin' in df_processed.columns else np.nan
                        
                        df_processed.at[idx, var] = self._resolve_median_fallback(gender, agebin, var)
        
        if self.impute_missing:
            df_processed = self._impute_dataframe(df_processed)

        # Transforms & Scalers Execution
        if hasattr(self, 'highly_skewed_'):
            present_skewed = [v for v in self.highly_skewed_ if v in df_processed.columns]
            if self.highly_skewed_ and present_skewed:
                pt_out = self.power_transformer_.transform(df_processed[present_skewed])
                df_processed[present_skewed] = self.robust_scaler_.transform(pt_out)
                
        if hasattr(self, 'normal_vars_'):
            present_normal = [v for v in self.normal_vars_ if v in df_processed.columns]
            if self.normal_vars_ and present_normal:
                df_processed[present_normal] = self.standard_scaler_.transform(df_processed[present_normal])

        # Execute consistent fast one-hot-encoding
        present_cat = [v for v in self.cat_vars if v in df_processed.columns]
        if present_cat:
            df_processed = pd.get_dummies(df_processed, columns=present_cat, drop_first=True)
            
        if hasattr(self, 'dummy_columns_'):
            missing_cols = set(self.dummy_columns_) - set(df_processed.columns)
            for c in missing_cols:
                df_processed[c] = 0
            # Force strict column alignment with original fitted training pipeline
            valid_out_cols = [c for c in self.dummy_columns_ if c in df_processed.columns]
            df_processed = df_processed[valid_out_cols]
        
        return df_processed
