import os
import ollama
import requests
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
print("\n|INFO| MASTER FILE columns:\n", master_file_columns)
print("|INFO| MASTER FILE head:\n", master_file.head())


# TRANSACTION FILE loading and preview
try:
    transaction_file = pd.read_excel("dataset/NP_NI_Cross-Re_2024-12.xlsx", sheet_name="aug-24")
except FileNotFoundError:
    print("\n|ERROR| TRANSACTION FILE not found")
    exit()

transaction_file_columns = transaction_file.columns.tolist()
print("\n|INFO| TRANSACTION FILE columns:\n", transaction_file_columns)
print("|INFO| TRANSACTION FILE head:\n", transaction_file.head())


def column_exists(df, col):
    """Checks if a column exists in a DataFrame."""
    if col not in df.columns:
        print(f"|WARNING| Column '{col}' missing in master file. Skipping this pass.")
        return False
    return True


def save_pass_df(df, pass_name, entry_index):
    """Saves a DataFrame for a specific pass and entry."""
    file_path = os.path.join(TEMP_DIR, f"pass_{entry_index}_{pass_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"|INFO| Saved {len(df)} rows to {file_path}")


def process_transaction_entry(entry_index):
    """
    Processes a single entry from the transaction file to find matching
    master file item codes using an LLM.
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
        new_filtered = []
        for idx, row in filtered_df.iterrows():
            try:
                row_value = str(row['catcode']) if pd.notna(row['catcode']) else ""
                prompt = f"""
Search value: {search_catcode1}
Row catcode: {row_value}

Rules:
- The 'Search value' is a numerical category code.
- The 'Row catcode' is a numerical category code.
- Both values must be an exact match to be considered 'true'.
- Ignore leading or trailing whitespace.
- Only reply with 'true' or 'false'.
"""
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    options={'temperature': 0.2, 'top_p': 0.1}
                )
                if "true" in response["message"]["content"].lower():
                    new_filtered.append(row)
            except Exception as e:
                print(f"|ERROR| catcode match failed for row {idx}: {e}")
        
        filtered_df = pd.DataFrame(new_filtered, columns=master_file.columns)
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After catcode filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "catcode", entry_index)

    # Pass 2: company (partial and case-insensitive match)
    if column_exists(filtered_df, 'company'):
        new_filtered = []
        for idx, row in filtered_df.iterrows():
            try:
                row_value = str(row['company']) if pd.notna(row['company']) else ""
                prompt = f"""
Search value: {search_company1}
Row company: {row_value}

Rules:
- The 'Search value' must be an exact match or contained within the 'Row company' value.
- Ignore differences in case (e.g., 'APPLE' matches 'apple').
- Only reply with 'true' or 'false'.
"""
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    options={'temperature': 0.2, 'top_p': 0.1}
                )
                if "true" in response["message"]["content"].lower():
                    new_filtered.append(row)
            except Exception as e:
                print(f"|ERROR| company match failed for row {idx}: {e}")
        
        filtered_df = pd.DataFrame(new_filtered, columns=master_file.columns)
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After company filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "company", entry_index)

    # Pass 3: brand (partial and case-insensitive match)
    if column_exists(filtered_df, 'brand'):
        new_filtered = []
        for idx, row in filtered_df.iterrows():
            try:
                row_value = str(row['brand']) if pd.notna(row['brand']) else ""
                prompt = f"""
Search value: {search_brand1}
Row brand: {row_value}

Rules:
- The 'Search value' must be an exact or partial match to the 'Row brand' value.
- Ignore differences in case (e.g., 'APPLE' matches 'apple').
- The match should be primarily an EXACT MATCH or, a partial, case-insensitive string match where the 'Search value' is present wholly within the 'Row brand' value.
- Only reply with 'true' or 'false'.
"""
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    options={'temperature': 0.2, 'top_p': 0.1}
                )
                if "true" in response["message"]["content"].lower():
                    new_filtered.append(row)
            except Exception as e:
                print(f"|ERROR| brand match failed for row {idx}: {e}")
        
        filtered_df = pd.DataFrame(new_filtered, columns=master_file.columns)
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After brand filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "brand", entry_index)

    # Pass 4: packtype (exact and case-insensitive match)
    if column_exists(filtered_df, 'packtype'):
        new_filtered = []
        for idx, row in filtered_df.iterrows():
            try:
                row_value = str(row['packtype']) if pd.notna(row['packtype']) else ""
                prompt = f"""
Search value: {search_packtype1}
Row packtype: {row_value}

Rules:
- The 'Search value' and 'Row packtype' must match exactly.
- Ignore differences in case.
- Only reply with 'true' or 'false'.
"""
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    options={'temperature': 0.2, 'top_p': 0.1}
                )
                if "true" in response["message"]["content"].lower():
                    new_filtered.append(row)
            except Exception as e:
                print(f"|ERROR| packtype match failed for row {idx}: {e}")

        filtered_df = pd.DataFrame(new_filtered, columns=master_file.columns)
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
        print(f"|INFO| After packtype filter: {len(filtered_df)} rows remain")
        save_pass_df(filtered_df, "packtype", entry_index)

    # Pass 5: qty + uom (exact numerical match for qty, exact case-insensitive match for uom)
    if column_exists(filtered_df, 'qty') and column_exists(filtered_df, 'uom'):
        new_filtered = []
        for idx, row in filtered_df.iterrows():
            try:
                row_qty = str(row['qty']) if pd.notna(row['qty']) else ""
                row_uom = str(row['uom']) if pd.notna(row['uom']) else ""
                prompt = f"""
Search qty: {search_qty1}
Row qty: {row_qty}

Search uom: {search_uom1}
Row uom: {row_uom}

Rules:
- 'Search qty' and 'Row qty' must be an exact numerical match.
- 'Search uom' and 'Row uom' must be an exact, case-insensitive string match.
- Both conditions must be met for a 'true' response.
- Only reply with 'true' or 'false'.
"""
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    options={'temperature': 0.2, 'top_p': 0.1}
                )
                if "true" in response["message"]["content"].lower():
                    new_filtered.append(row)
            except Exception as e:
                print(f"|ERROR| qty+uom match failed for row {idx}: {e}")

        filtered_df = pd.DataFrame(new_filtered, columns=master_file.columns)
        if not filtered_df.empty:
            last_successful_df = filtered_df.copy()
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


# --- Main processing loop ---
all_results = []
for entry_index in range(len(transaction_file)):
    result_df = process_transaction_entry(entry_index)
    all_results.append(result_df)


# Concatenate all individual results into a single DataFrame
FINAL_OUTPUT = pd.concat(all_results, ignore_index=True)


# Save and print the final output
final_output_path = os.path.join(OUTPUT_DIR, "FINAL_OUTPUT.csv")
FINAL_OUTPUT.to_csv(final_output_path, index=False)
print(f"\n|OUTPUT| Final results saved to: {final_output_path}")

print("\n|OUTPUT| Final Output DataFrame:")
print(FINAL_OUTPUT)