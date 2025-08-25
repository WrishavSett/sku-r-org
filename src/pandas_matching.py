import os
import re
import shutil
import numpy as np
import pandas as pd


# Create temporary and output folders
os.makedirs("./temp", exist_ok=True)
os.makedirs("./output", exist_ok=True)


# CONFIGURATION
TEMP_DIR = "./temp"
OUTPUT_DIR = "./output"


# MASTER FILE loading and preview
try:
    master_file = pd.read_excel("dataset/NP_ItemMaster_Detailed_2025-07.xlsx")
except FileNotFoundError:
    print("\n|ERROR| MASTER FILE not found")
    exit()

master_file_columns = master_file.columns.tolist()
# print("\n|INFO| MASTER FILE columns:\n", master_file_columns)
# print("|INFO| MASTER FILE head:\n", master_file.head())


# TRANSACTION FILE loading and preview
try:
    # sheet_names = pd.ExcelFile("dataset/NP_NI_Cross-Re_2024-12.xlsx").sheet_names
    sheet_name = "dec-24"
    transaction_file = pd.read_excel("dataset/NP_NI_Cross-Re_2024-12.xlsx", sheet_name=sheet_name)
except FileNotFoundError:
    print("\n|ERROR| TRANSACTION FILE not found")
    exit()

transaction_file_columns = transaction_file.columns.tolist()
# print("\n|INFO| TRANSACTION FILE columns:\n", transaction_file_columns)
# print("|INFO| TRANSACTION FILE head:\n", transaction_file.head())


# --- Helpers ---
def column_exists(df, col):
    """Checks if a column exists in a DataFrame."""
    if col not in df.columns:
        print(f"|WARNING| Column '{col}' missing in master file. Skipping this pass.")
        return False
    return True


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = text.replace("–", "-").replace("—", "-")  # normalize dashes
    text = text.replace("\\", "/")  # optional: unify backslashes
    return text


def save_pass_df(df, pass_name, entry_index):
    """Saves a DataFrame for a specific pass and entry."""
    file_path = os.path.join(TEMP_DIR, f"pass_{entry_index}_{pass_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"|INFO| Saved {len(df)} rows to {file_path}")


for col in ['catcode', 'company', 'brand', 'packtype', 'uom', 'itemcode']:
    if col in master_file.columns:
        master_file[col] = master_file[col].map(normalize_text)

for col in ['CATEGORY', 'MANUFACTURE', 'BRAND', 'PACKTYPE', 'PACKSIZE', 'ITEMDESC', 'ITEMCODE']:
    if col in transaction_file.columns:
        transaction_file[col] = transaction_file[col].map(normalize_text)


def process_transaction_entry(entry_index):
    """
    Processes a single entry from the transaction file to find matching
    master file item codes.
    """
    
    # Query input from the transaction file for the current entry
    search_itemcode1 = str(transaction_file['ITEMCODE'][entry_index])
    search_catcode1 = str(transaction_file['CATEGORY'][entry_index])
    search_company1 = str(transaction_file['MANUFACTURE'][entry_index])
    search_brand1 = str(transaction_file['BRAND'][entry_index])
    search_packtype1 = str(transaction_file['PACKTYPE'][entry_index])
    search_base_pack1 = str(transaction_file['PACKSIZE'][entry_index])
    search_qty1 = ''.join([char for char in search_base_pack1 if char.isdigit()])
    search_uom1 = ''.join([char for char in search_base_pack1 if char.isalpha()])
    search_itemdesc1 = str(transaction_file['ITEMDESC'][entry_index])

    print(f"""\n|INFO| Search values (from transaction file) for entry '{entry_index}':
          ITEMCODE: {search_itemcode1}
          CATCODE: {search_catcode1}
          COMPANY: {search_company1}
          BRAND: {search_brand1}
          PACKTYPE: {search_packtype1}
          BASE PACK: {search_base_pack1}
          QTY: {search_qty1}
          UOM: {search_uom1}
          ITEMDESC: {search_itemdesc1}""")

    filtered_df = master_file.copy()
    last_successful_df = pd.DataFrame()

    # Pass 1: catcode (exact match)
    if column_exists(filtered_df, 'catcode'):
        filtered_df = filtered_df[filtered_df['catcode'].astype(str).str.strip() == search_catcode1.strip()]
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After catcode filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "catcode", entry_index)

    # Pass 2: company (partial and case-insensitive match)
    if column_exists(filtered_df, 'company'):
        filtered_df = filtered_df[filtered_df['company'].astype(str).str.contains(search_company1, case=False, na=False)]
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After company filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "company", entry_index)

    # Pass 3: brand (partial and case-insensitive match)
    if column_exists(filtered_df, 'brand'):
        filtered_df = filtered_df[filtered_df['brand'].astype(str).str.contains(search_brand1, case=False, na=False)]
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After brand filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "brand", entry_index)

    # Pass 4: packtype (exact and case-insensitive match)
    if column_exists(filtered_df, 'packtype'):
        filtered_df = filtered_df[filtered_df['packtype'].astype(str).str.lower().str.strip() == search_packtype1.lower().strip()]
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After packtype filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "packtype", entry_index)

    # Pass 5: qty + uom (exact numerical match for qty, exact case-insensitive match for uom)
    if column_exists(filtered_df, 'qty') and column_exists(filtered_df, 'uom'):
        try:
            search_qty_num = pd.to_numeric(search_qty1)
            qty_mask = filtered_df['qty'].astype(float) == search_qty_num
            uom_mask = filtered_df['uom'].astype(str).str.lower().str.strip() == search_uom1.lower().strip()
            filtered_df = filtered_df[qty_mask & uom_mask]
            if not filtered_df.empty:
                last_successful_df = filtered_df.copy()
        except ValueError:
            print(f"|WARNING| Could not convert search_qty1 ('{search_qty1}') to a number. Skipping qty+uom pass.")
        
        print(f"|INFO| After qty+uom filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "qty_uom", entry_index)


    # Final output generation for this entry
    if not last_successful_df.empty:
        m_itemcodes = last_successful_df['itemcode'].unique().tolist()
        m_itemcodes_str = ' || '.join([str(item) for item in m_itemcodes])
        
        return pd.DataFrame({
            't_itemcode': [search_itemcode1],
            'm_itemcode(s)': [m_itemcodes_str]
        })
    
    return pd.DataFrame({
        't_itemcode': [search_itemcode1],
        'm_itemcode(s)': [None]
    })


def clear_temp_dir():
    """Remove all contents inside the TEMP_DIR but keep the folder."""
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subdirectory
        except Exception as e:
            print(f"|WARNING| Failed to delete {file_path}. Reason: {e}")


def main():
    # --- Main processing loop ---
    all_results = []
    for entry_index in range(len(transaction_file)):
        result_df = process_transaction_entry(entry_index)
        all_results.append(result_df)


    # Concatenate all individual results into a single DataFrame
    FINAL_OUTPUT = pd.concat(all_results, ignore_index=True)


    # Save and print the final output
    final_output_path = os.path.join(OUTPUT_DIR, f"output_{sheet_name}.csv")
    FINAL_OUTPUT.to_csv(final_output_path, index=False)
    print(f"\n|OUTPUT| Final results saved to: {final_output_path}")

    print("\n|OUTPUT| Final Output DataFrame:")
    print(FINAL_OUTPUT)

    clear_temp_dir()
    print(f"\n|CLEANUP| Temp files removed from {TEMP_DIR}")


if __name__ == "__main__":
    main()