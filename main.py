import os
import ollama
import requests
import numpy as np
import pandas as pd


# temporary folder creation
os.makedirs("./temp", exist_ok=True)


# CONFIGURATION
MODEL_NAME = "llama3.2:3b"
ENTRY_INDEX = 31
TEMP_DIR = "./temp"


# MASTER FILE loading and preview
try:
    master_file = pd.read_excel("dataset/NP_ItemMaster_Detailed_2025-07.xlsx")
except FileNotFoundError:
    print("\n|ERROR| MASTER FILE not found")

master_file_columns = master_file.columns.tolist()
print("\n|INFO| MASTER FILE columns:\n", master_file_columns)
print("|INFO| MASTER FILE head:\n", master_file.head())


# TRANSACTION FILE loading and preview
try:
    transaction_file = pd.read_excel("dataset/NP_NI_Cross-Re_2024-12.xlsx", sheet_name="aug-24")
except FileNotFoundError:
    print("\n|ERROR| TRANSACTION FILE not found")

transaction_file_columns = transaction_file.columns.tolist()
print("\n|INFO| TRANSACTION FILE columns:\n", transaction_file_columns)
print("|INFO| TRANSACTION FILE head:\n", transaction_file.head())


# Query input
entry = 31
search_catcode1 = str(transaction_file['CATEGORY'][entry])
    # "catcode" corresponds to "CATEGORY"
search_company1 = str(transaction_file['MANUFACTURE'][entry])
    # "company" corresponds to "MANUFACTURE"
search_brand1 = str(transaction_file['BRAND'][entry])
    # "brand" corresponds to "BRAND"
search_packtype1 = str(transaction_file['PACKTYPE'][entry])
    # "packtype" corresponds to "PACKTYPE"
search_base_pack1 = str(transaction_file['PACKSIZE'][entry])
    # "base_pack" corresponds to "PACKSIZE"
search_qty1 = ''.join([char for char in search_base_pack1 if char.isdigit()])
    # qty is for master_file correspondance
search_uom1 = ''.join([char for char in search_base_pack1 if char.isalpha()])
    # uom is for master_file correspondance
search_itemdesc1 = str(transaction_file['ITEMDESC'][entry])
    # itemdesc is for transaction_file correspondance for field "ITEMDESC"


print(f"""\n|INFO| Search values (from transaction file) for entry '{entry}':
      CATCODE: {search_catcode1}
      COMPANY: {search_company1}
      BRAND: {search_brand1}
      PACKTYPE: {search_packtype1}
      BASE PACK: {search_base_pack1}
      QTY: {search_qty1}
      UOM: {search_uom1}
      ITEMDESC: {search_itemdesc1}""")


filtered_df = master_file.copy()

def column_exists(df, col):
    if col not in df.columns:
        print(f"|WARNING| Column '{col}' missing in master file. Skipping this pass.")
        return False
    return True

def save_pass_df(df, pass_name):
    file_path = os.path.join("temp", f"{pass_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"|INFO| Saved {len(df)} rows to {file_path}")


# Pass 1: catcode
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
                options={'temperature': 0.0, 'top_p': 0.1}
            )
            if "true" in response["message"]["content"].lower():
                new_filtered.append(row)
        except Exception as e:
            print(f"|ERROR| catcode match failed for row {idx}: {e}")

    filtered_df = pd.DataFrame(new_filtered)
    print(f"\n|INFO| After catcode filter: {len(filtered_df)} rows remain")
    save_pass_df(filtered_df, "catcode")


# Pass 2: company
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
- The match should be primarily an EXACT MATCH or, a partial, case-insensitive string match where the 'Search value' is present wholly within the 'Row company' value.
- Only reply with 'true' or 'false'.
"""
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": "You are a highly logical and precise data comparison tool. Your only function is to determine if two values match based on a strict set of rules. You will only respond with the exact word 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                options={'temperature': 0.0, 'top_p': 0.1}
            )
            if "true" in response["message"]["content"].lower():
                new_filtered.append(row)
        except Exception as e:
            print(f"|ERROR| company match failed for row {idx}: {e}")

    filtered_df = pd.DataFrame(new_filtered)
    print(f"|INFO| After company filter: {len(filtered_df)} rows remain")
    save_pass_df(filtered_df, "company")


# Pass 3: brand
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
                options={'temperature': 0.0, 'top_p': 0.1}
            )
            if "true" in response["message"]["content"].lower():
                new_filtered.append(row)
        except Exception as e:
            print(f"|ERROR| brand match failed for row {idx}: {e}")

    filtered_df = pd.DataFrame(new_filtered)
    print(f"|INFO| After brand filter: {len(filtered_df)} rows remain")
    save_pass_df(filtered_df, "brand")


# Pass 4: packtype
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
                options={'temperature': 0.0, 'top_p': 0.1}
            )
            if "true" in response["message"]["content"].lower():
                new_filtered.append(row)
        except Exception as e:
            print(f"|ERROR| packtype match failed for row {idx}: {e}")

    filtered_df = pd.DataFrame(new_filtered)
    print(f"|INFO| After packtype filter: {len(filtered_df)} rows remain")
    save_pass_df(filtered_df, "packtype")


# Pass 5: qty + uom
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
                options={'temperature': 0.0, 'top_p': 0.1}
            )
            if "true" in response["message"]["content"].lower():
                new_filtered.append(row)
        except Exception as e:
            print(f"|ERROR| qty+uom match failed for row {idx}: {e}")

    filtered_df = pd.DataFrame(new_filtered)
    print(f"|INFO| After qty+uom filter: {len(filtered_df)} rows remain")
    save_pass_df(filtered_df, "qty_uom")


# Final output
print("\n|INFO| Final filtered dataframe:")
print(filtered_df)