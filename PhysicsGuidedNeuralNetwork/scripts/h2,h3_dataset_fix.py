import pandas as pd
import os
import glob
import re

def process_dataset_by_h_value(input_dir, output_dir):
    """
    Process CSV files based on h value extracted from filename:
    - h2: Drop columns TC5-TC10 only
    - h3: Reorder columns and remove TC8,9,10 and specific columns
    - h6: Keep unchanged (perfect)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Maps for reference
    h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
    flux_map = {88: 25900, 78: 21250, 73: 19400}
    abs_map = {0: 3, 92: 100}
    surf_map = {0: 0.98, 1: 0.76}
    
    # Pattern to extract h value from filename
    pattern = r"h(\d+)_flux(\d+)_abs(\d+)(?:_[A-Za-z0-9]+)*_surf([01])(?:_[A-Za-z0-9]+)*[\s_]+(\d+)s\b"
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            # Extract h value from filename
            match = re.search(pattern, filename)
            if not match:
                print(f"Skipping {filename}: Could not extract h value from filename.")
                continue
            
            h_value = int(match.group(1))
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Process based on h value
            if h_value == 2:
                # h2: Drop columns TC5-TC10 only, keep all others
                columns_to_drop = ['TC4','TC5', 'TC6', 'TC7', 'TC8', 'TC9','TC_9_5','TC10']
                df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
                print(f"h2 - Processed {filename}: Dropped TC5-TC10")
                
            elif h_value == 3:
                # h3: First reorder columns, then remove specific columns
                
                # Define the correct column order
                correct_order = [
                    'Time', 'TC1_tip', 'TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7', 'TC8', 'TC9', 'TC10',
                    'Theoretical_Temps_1', 'Theoretical_Temps_2', 'Theoretical_Temps_3', 'Theoretical_Temps_4',
                    'Theoretical_Temps_5', 'Theoretical_Temps_6', 'Theoretical_Temps_7', 'Theoretical_Temps_8',
                    'Theoretical_Temps_9', 'Theoretical_Temps_10', 'Theoretical_Temps_11',
                    'h', 'flux', 'abs', 'surf'
                ]
                
                # Reorder columns (only include columns that exist in the dataframe)
                existing_columns = [col for col in correct_order if col in df.columns]
                df_reordered = df[existing_columns]
                
                # Remove specific columns for h3
                columns_to_remove = ['TC6','TC7','TC8', 'TC9', 'TC10', 'TC_Bottom_rec_groove', 'TC_wall_ins_ext', 'TC_bottom_ins_groove']
                df_processed = df_reordered.drop(columns=[col for col in columns_to_remove if col in df_reordered.columns])
                print(f"h3 - Processed {filename}: Reordered columns and removed TC8,9,10 and specific columns")
                
            elif h_value == 6:
                # h6: Perfect, no changes needed
                df_processed = df.copy()
                print(f"h6 - Processed {filename}: No changes needed (perfect)")
                
            else:
                print(f"Skipping {filename}: Unsupported h value {h_value}")
                continue
            
            # Save processed file
            output_path = os.path.join(output_dir, filename)
            df_processed.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# === Usage ===
cleaned_input_dir = "PhysicsGuidedNeuralNetwork/data/new_processed_reset"
time_reset_output_dir = "PhysicsGuidedNeuralNetwork/data/new_processed_fix_new"

process_dataset_by_h_value(cleaned_input_dir, time_reset_output_dir)