import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

STATIC_VARS = [
    'Age',
    'Gender',
    'Height',
    'Weight',
    'ICUType',
    'RecordID',
]


def process_physionet_set(set_name, input_base_path, output_base_path):
    """
    Processes a single PhysioNet 2012 set, pivots variables dynamically, 
    and saves the result as a Parquet file.
    """
    # Convert string paths to Path objects if they aren't already
    input_base_path = Path(input_base_path)
    output_base_path = Path(output_base_path)
    
    patient_dir = input_base_path / f"set-{set_name}"
    outcome_path = input_base_path / f"Outcomes-{set_name}.txt"
    
    # Load outcomes safely
    if not outcome_path.exists():
        print(f"Warning: Outcome file {outcome_path} not found. Labels will be skipped.")
        outcomes = pd.DataFrame()
    else:
        outcomes = pd.read_csv(outcome_path).set_index('RecordID')
    
    all_patient_data = []
    
    print(f"--- Processing Set: {set_name} ---")


    for file_path in patient_dir.glob("*.txt"):
        raw_df = pd.read_csv(file_path)
        patient_id = int(file_path.stem)
        
        raw_df['Hours'] = raw_df['Time'].apply(
            lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60
        )
        raw_df['Hour_Bin'] = np.ceil(raw_df['Hours']).astype(int)

        grid_df = pd.DataFrame({'Timestamp': range(49)})
        grid_df['PatientID'] = patient_id

        for var in STATIC_VARS:
            val = raw_df[raw_df['Parameter'] == var]['Value'].unique()
            grid_df[var] = val[0] if len(val) > 0 else np.nan

        dynamic_obs = raw_df[~raw_df['Parameter'].isin(STATIC_VARS)]
        pivoted = dynamic_obs.pivot_table(
            index='Hour_Bin',
            columns='Parameter',
            values='Value',
            aggfunc='last'
        )

        final_patient_df = grid_df.merge(
            pivoted, left_on='Timestamp', right_index=True, how='left'
        )

        if not outcomes.empty and patient_id in outcomes.index:
            final_patient_df['Label'] = outcomes.loc[patient_id, 'In-hospital_death']
            
        all_patient_data.append(final_patient_df)
        
    # Combine and Sort Columns Alphabetically, keeping Timestamp first
    combined_df = pd.concat(all_patient_data, ignore_index=True)
    cols = sorted([c for c in combined_df.columns if c != 'Timestamp'])
    combined_df = combined_df[['Timestamp'] + cols]
    
    # Print variables
    print(f"Total variables found: {len(combined_df.columns)}")
    print(f"Columns: {', '.join(combined_df.columns.tolist())}\n")
    
    # Save the result
    output_base_path.mkdir(parents=True, exist_ok=True)
    save_path = output_base_path / f"processed_set_{set_name}.parquet"
    combined_df.to_parquet(save_path, engine='fastparquet')
    
    return combined_df 

