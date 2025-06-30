import pathlib
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import latency_model
import power_model
import energy_model

def parse_question_inputs(file_path: pathlib.Path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def parse_xlsx_data(file_path: pathlib.Path):
    """
    Parse Excel file and extract specified columns from all sheets.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary with sheet names as keys and DataFrames with columns: 
        subject, question_id, input_tokens, output_tokens, sub_id
    """
    try:
        # Read all sheets of the Excel file
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        
        result_dict = {}
        required_columns = ["subject", "question_id", "input_tokens", "output_tokens"]
        
        for sheet_name, df in all_sheets.items():
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in sheet '{sheet_name}' of {file_path}: {missing_columns}")
                print(f"Available columns in sheet '{sheet_name}': {list(df.columns)}")
                continue
            
            # Select only the required columns
            extracted_data = df[required_columns].copy()
            # Add sub_id column starting from 0 for each subject
            extracted_data['sub_id'] = extracted_data.groupby('subject').cumcount()
            
            name_dict = {
                "DeepSeek-R1-Distill-Llama-8B": "DSR1-Llama-8B",
                "DeepSeek-R1-Distill-Qwen-1_5B": "DSR1-Qwen-1.5B",
                "DeepSeek-R1-Distill-Qwen-14B": "DSR1-Qwen-14B",
                "L1-Qwen-1_5B-Max": "L1-Qwen-1.5B-Max",
            }
            result_dict[name_dict[sheet_name]] = extracted_data
        
        return result_dict
        
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None


def parse_old_xlsx_data(file_path: pathlib.Path):
    """
    Parse old format Excel file - sheet 0, extract subset and output_tokens,
    rename subset to subject, and add sub_id for same subject names.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with columns: subject, output_tokens, sub_id
    """
    try:
        # Read sheet 0 of the Excel file
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Check if required columns exist
        required_columns = ["subset", "output_tokens"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in {file_path}: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Select only the required columns
        extracted_data = df[required_columns].copy()
        
        # Rename 'subset' to 'subject'
        extracted_data = extracted_data.rename(columns={'subset': 'subject'})
        
        # Add sub_id column starting from 0 for each subject
        extracted_data['sub_id'] = extracted_data.groupby('subject').cumcount()
        
        return extracted_data
        
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    data_dir = pathlib.Path("dataset")
    summary_xlsx_path = data_dir / "full_mmlu_by_model_tegra.xlsx"
    extracted_data = parse_xlsx_data(summary_xlsx_path)
    
    if extracted_data:
        print(f"Successfully parsed {len(extracted_data)} sheets:")
        for sheet_name, df in extracted_data.items():
            print(f"\nSheet: {sheet_name}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head())
    else:
        print("Failed to parse the Excel file")
    
    # Parse old format spreadsheets
    old_data = {}
    subdata_dir = data_dir / "dataset"
    for file_path in subdata_dir.glob("*.xlsx"):
        sub_id = file_path.stem
        sub_data = parse_old_xlsx_data(file_path)
        if sub_data is not None:
            old_data[sub_id] = sub_data
    
    # Look up input_tokens from extracted_data for each old_data spreadsheet
    if extracted_data and old_data:
        print(f"\nLooking up input_tokens for {len(old_data)} old format files...")
        
        for old_file_id, old_df in old_data.items():
            print(f"\nProcessing {old_file_id}:")
            
            # Try to find matching sheet in extracted_data
            # Look for sheets that might contain the data for this file
            matching_sheet = None
            for sheet_name, sheet_df in extracted_data.items():
                # Check if this sheet has the required columns for lookup
                if "subject" in sheet_df.columns and "sub_id" in sheet_df.columns and "input_tokens" in sheet_df.columns:
                    # Check if there's any overlap in subjects
                    old_subjects = set(old_df['subject'].unique())
                    sheet_subjects = set(sheet_df['subject'].unique())
                    if old_subjects.intersection(sheet_subjects):
                        matching_sheet = sheet_name
                        break
            
            if matching_sheet:
                print(f"  Found matching sheet: {matching_sheet}")
                reference_df = extracted_data[matching_sheet]
                
                # Merge old_df with reference_df to get input_tokens
                merged_df = old_df.merge(
                    reference_df[['subject', 'sub_id', 'input_tokens']], 
                    on=['subject', 'sub_id'], 
                    how='left'
                )
                
                # Check how many matches were found
                matched_count = merged_df['input_tokens'].notna().sum()
                total_count = len(merged_df)
                print(f"  Matched {matched_count}/{total_count} rows with input_tokens")
                
                # Show sample of merged data
                print(f"  Sample of merged data:")
                print(merged_df.head())
                
                # Update old_data with the merged dataframe
                old_data[old_file_id] = merged_df
            else:
                print(f"  No matching sheet found for {old_file_id}")
                print(f"  Available sheets: {list(extracted_data.keys())}")
                print(f"  Subjects in {old_file_id}: {list(old_df['subject'].unique())}")
    
    # Merge all old data if any was found
    if old_data:
        print(f"\nMerged data from {len(old_data)} old format files:")
        for sub_id, df in old_data.items():
            print(f"  {sub_id}: {df.shape}")
    else:
        print("No old format files found or successfully parsed")


    supported_models = latency_model.prefill_coeffs.keys()

    def add_latency_power_energy_to_df(df, sub_id):
        supported_model = None
        for model_name in supported_models:
            if model_name in sub_id:
                supported_model = model_name
                break
        if supported_model:
            for index, row in df.iterrows():
                input_length = row['input_tokens']
                output_length = row['output_tokens']
                
                # Calculate latency
                prefill_latency, decode_latency, total_latency = latency_model.total_latency_model(supported_model, input_length, output_length)
                df.at[index, 'prefill_latency'] = prefill_latency
                df.at[index, 'decode_latency'] = decode_latency
                df.at[index, 'total_latency'] = total_latency
                
                # Calculate power
                prefill_power, decode_power = power_model.total_power_model(supported_model, input_length, output_length)
                df.at[index, 'prefill_power'] = prefill_power
                df.at[index, 'decode_power'] = decode_power
                
                # Calculate energy
                prefill_energy, decode_energy, total_energy = energy_model.total_energy_model(supported_model, input_length, output_length)
                df.at[index, 'prefill_energy'] = prefill_energy
                df.at[index, 'decode_energy'] = decode_energy
                df.at[index, 'total_energy'] = total_energy
                
            print(f"  {sub_id} is supported model {supported_model}")
        else:
            print(f"  {sub_id} is not supported model")


    for sub_id, df in old_data.items():
        add_latency_power_energy_to_df(df, sub_id)

    for sub_id, df in extracted_data.items():
        add_latency_power_energy_to_df(df, sub_id)

    def parse_tokens_latency_power_energy(df, sub_id):
        if 'prefill_latency' not in df.columns or 'decode_latency' not in df.columns:
            print(f"Skipping {sub_id} - no latency data (unsupported model)")
            return None
            
        sum_input_tokens = 0
        sum_output_tokens = 0
        sum_input_latency = 0
        sum_output_latency = 0
        sum_input_power = 0
        sum_output_power = 0
        sum_input_energy = 0
        sum_output_energy = 0

        for index, row in df.iterrows():
            sum_input_tokens += row['input_tokens']
            sum_output_tokens += row['output_tokens']
            sum_input_latency += row['prefill_latency']
            sum_output_latency += row['decode_latency']
            sum_input_power += row['prefill_power']
            sum_output_power += row['decode_power']
            sum_input_energy += row['prefill_energy']
            sum_output_energy += row['decode_energy']
        
        total_latency = sum_input_latency + sum_output_latency
        total_energy = sum_input_energy + sum_output_energy
        
        print(f"\n=== Summary for {sub_id} ===")
        print(f"Total Input Tokens: {sum_input_tokens:,}")
        print(f"Total Output Tokens: {sum_output_tokens:,}")
        print(f"Total Input Latency: {sum_input_latency:.2f} seconds")
        print(f"Total Output Latency: {sum_output_latency:.2f} seconds")
        print(f"Average Input Power: {sum_input_power/len(df):.2f} watts")
        print(f"Average Output Power: {sum_output_power/len(df):.2f} watts")
        print(f"Total Input Energy: {sum_input_energy:.2f} joules")
        print(f"Total Output Energy: {sum_output_energy:.2f} joules")
        print(f"Total Energy: {total_energy:.2f} joules")
        
        tokens_ratio =  sum_output_tokens / sum_input_tokens
        latency_ratio = sum_output_latency / sum_input_latency
        energy_ratio = sum_output_energy / sum_input_energy
        
        print(f"Output-to-Input Tokens Ratio: {tokens_ratio:.2f}")
        print(f"Output-to-Input Latency Ratio: {latency_ratio:.2f}")
        print(f"Output-to-Input Energy Ratio: {energy_ratio:.2f}")
        
        return {
            'input_tokens': sum_input_tokens,
            'output_tokens': sum_output_tokens,
            'input_latency': sum_input_latency,
            'output_latency': sum_output_latency,
            'input_energy': sum_input_energy,
            'output_energy': sum_output_energy,
            'total_energy': total_energy
        }

    # Calculate sums for old data (supported models only)
    model_names = ['DSR1-Llama-8B', 'DSR1-Qwen-1.5B', 'DSR1-Qwen-14B']
    for model_name in model_names:
        if model_name in extracted_data:
            df = extracted_data[model_name] 
            parse_tokens_latency_power_energy(df, model_name)

    # Calculate sums for extracted data
    extracted_data_sums = {}
    for sub_id, df in extracted_data.items():
        result = parse_tokens_latency_power_energy(df, sub_id)
        if result is not None: 
            extracted_data_sums[sub_id] = result

    # Save old_data to CSV files
    csv_dir = pathlib.Path("outputs")
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)

    if old_data:
        print(f"\nSaving old format data to CSV files...")
        for sub_id, df in old_data.items():
            csv_filename = f"old_format_{sub_id}.csv"
            
            df.to_csv(csv_dir/ csv_filename, index=False)
            print(f"  Saved {sub_id} to {csv_filename} ({df.shape[0]} rows)")
    else:
        print("No old format data to save")

    if extracted_data:
        print(f"\nSaving old format data to CSV files...")
        for sub_id, df in extracted_data.items():
            # Create filename for the old format data
            csv_filename = f"new_format_{sub_id}.csv"
            
            # Save to CSV
            df.to_csv(csv_dir/ csv_filename, index=False)
            print(f"  Saved {sub_id} to {csv_filename} ({df.shape[0]} rows)")
    else:
        print("No new format data to save")
