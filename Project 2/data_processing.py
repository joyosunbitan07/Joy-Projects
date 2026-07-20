import glob
import pandas as pd
from sklearn.model_selection import train_test_split


# Making the  function that processes a single raw text line
def parse_libsvm_line(line):
    parts = line.strip().split(" ")
    raw_risk_score = float(parts[0])
    feature_pairs = parts[1:]

    row_data = {}

    # Derive binary label
    if raw_risk_score >= 0.30:
        row_data["label"] = 1
    else:
        row_data["label"] = 0

    # Extract features
    for pair in feature_pairs:
        if ":" in pair:
            sub_parts = pair.split(":")
            feature_idx = sub_parts[0]
            feature_val = float(sub_parts[1])
            row_data[feature_idx] = feature_val

    return row_data


def main():
    # Looking for raw data inside a folder right next to the script
    folder_pattern = "./project2_raw_data/*.txt"
    file_list = glob.glob(folder_pattern)

    print(f"I have found {len(file_list)} files to process.")

    master_rows = []

    # Looping over every single text document found
    for file_path in file_list:
        print(f"Processing: {file_path}")

        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed_dict = parse_libsvm_line(line)
                master_rows.append(parsed_dict)

    print("\nAll files parsed successfully!")
    print(f"Total rows collected across all files: {len(master_rows)}")

    # Combine into one unified DataFrame
    master_df = pd.DataFrame(master_rows)
    # removing inconsistencies before I split
    master_df = master_df.drop_duplicates()
    print(f"Combined DataFrame shape: {master_df.shape}")

    # ========================================================
    ##  Rename Columns  Using Pandas Mapping (UPDATED CODE HERE) ##
    # ========================================================
    print("\n[!] Mapping features cleanly with proper types...")

    # Reference file loaded right from the same directory
    ref_df = pd.read_csv("./feature_name_to_number_mapping.csv")

    # Ensure everything is a string
    ref_df["feature_number_str"] = (
        ref_df["feature_number"].astype(str).str.strip()
    )
    ref_df["feature_name_str"] = ref_df["feature_name"].astype(str).str.strip()

    # Creating the explicit string-to-string dictionary look-up
    feature_map = dict(
        zip(ref_df["feature_number_str"], ref_df["feature_name_str"])
    )

    # Forcing master_df columns to be clean strings before mapping
    master_df.columns = master_df.columns.astype(str).str.strip()

    # Apply the renaming dictionary
    master_df = master_df.rename(columns=feature_map)

    print(f"Successfully mapped {len(feature_map)} feature headers.")

    # ========================================================
    ##  Data Cleaning ##
    # ========================================================
    # CRITICAL CORRECTION: In LibSVM/Sparse data format, an omitted feature
    # index does not mean the value is unknown (NaN). It means the API was called exactly 0.0 times.
    # We fill these gaps cleanly here to preserve the true signal.
    master_df = master_df.fillna(0.0)

    print("\n--- CLEANED FULL DATAFRAME SAMPLE ---")
    print(master_df.head())

    # ========================================================
    ##  Feature Creation (Calculated cleanly on Master Object) ##
    # ========================================================
    print("\n[!] Engineering structural interaction footprints...")
    #This line grabs only the raw columns and NOT the label column
    raw_api_cols = [col for col in master_df.columns if col != "label"]
    #updating these lines to calculate on raw_api_cols
    master_df["total_api_calls"] = master_df[raw_api_cols].sum(axis=1)
    master_df["unique_apis_used"] = (master_df[raw_api_cols] > 0).sum(axis=1)

    print("Features successfully expanded across master dataset configuration!")

    # ========================================================
    ##  Create Stratified Train / Test / Validate Splits ##
    # ========================================================
    # We use stratify=master_df['label'] to guarantee that the 90/10
    # population distribution is maintained identically across all 3 file segments.
    temp_train_df, test_df = train_test_split(
        master_df, test_size=0.20, random_state=42, stratify=master_df["label"]
    )

    # test_size=0.1875 inside an 80% chunk dynamically equals exactly 15% of the overall original dataset
    train_df, val_df = train_test_split(
        temp_train_df,
        test_size=0.1875,
        random_state=42,
        stratify=temp_train_df["label"],
    )

    print("\n--- STRATIFIED 3-WAY DATA SPLIT COMPLETE ---")
    print(f"Total training rows:   {len(train_df)}")
    print(f"Total validation rows: {len(val_df)}")
    print(f"Total testing rows:    {len(test_df)}")

    # ========================================================
    ##  Save Clean Processed Datasets to CSV ##
    # ========================================================
    print("\nSaving clean datasets with true feature properties...")

    # Save data copies flat into the current project2 directory
    train_df.to_csv("./train_flat.csv", index=False)
    val_df.to_csv("./val_flat.csv", index=False)
    test_df.to_csv("./test_flat.csv", index=False)

    print("!!! ALL DATA IS NOW PROCESSED WITH CORRECT SIGNALS AND EXPORTED !!!")


if __name__ == "__main__":
    main()