import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# Create temporary and output folders
os.makedirs("./temp", exist_ok=True)
os.makedirs("./output", exist_ok=True)


# CONFIGURATION
TEMP_DIR = "./temp"
OUTPUT_DIR = "./output"


# --- Configurable mappings ---
uom_mapping = {
    "ml": "ml", "millilitre": "ml",
    "l": "l", "litre": "l",
    "gm": "g", "gram": "g",
    "kg": "kg",
    "no": "pcs", "pcs": "pcs"
}

packtype_mapping = {
    "can": "can", "tin": "can", "tin c": "can",
    "pet": "bottle", "pbt": "bottle", "plbot": "bottle", "glbot": "bottle",
    "jar": "jar", "pljar": "jar", "gljar": "jar",
    "tpk": "tpk", "rgb": "rgb", "hl": "hl"
}


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
    # sheet_names = pd.ExcelFile("dataset/NP_NI_Cross-Re_2024-12.xlsx").sheet_names
    sheet_name = "aug-24"
    transaction_file = pd.read_excel("dataset/NP_NI_Cross-Re_2024-12.xlsx", sheet_name=sheet_name)
except FileNotFoundError:
    print("\n|ERROR| TRANSACTION FILE not found")
    exit()

transaction_file_columns = transaction_file.columns.tolist()
print("\n|INFO| TRANSACTION FILE columns:\n", transaction_file_columns)
print("|INFO| TRANSACTION FILE head:\n", transaction_file.head())


# --- Helpers ---
def normalize_value(value, mapping):
    """Normalize values using mapping dict."""
    if pd.isna(value):
        return None
    cleaned = str(value).strip().lower()
    return mapping.get(cleaned, cleaned)

def convert_qty_uom(qty, uom):
    """Convert qty+uom to base ML or G for strict comparison."""
    if pd.isna(qty) or pd.isna(uom):
        return None
    try:
        qty = float(qty)
    except:
        return None

    uom = str(uom).lower().strip()
    if uom in ["ml", "l"]:
        return qty * 1000 if uom == "l" else qty
    elif uom in ["g", "kg"]:
        return qty * 1000 if uom == "kg" else qty
    elif uom in ["pcs"]:
        return qty
    return None

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


# def process_transaction_entry(entry_index):
#     """
#     Processes a single entry from the transaction file to find matching
#     master file item codes.
#     """
    
#     # Query input from the transaction file for the current entry
#     search_itemcode1 = str(transaction_file['ITEMCODE'][entry_index])
#     search_catcode1 = str(transaction_file['CATEGORY'][entry_index])
#     search_company1 = str(transaction_file['MANUFACTURE'][entry_index])
#     search_brand1 = str(transaction_file['BRAND'][entry_index])
#     search_packtype1 = str(transaction_file['PACKTYPE'][entry_index])
#     search_base_pack1 = str(transaction_file['PACKSIZE'][entry_index])
#     search_qty1 = ''.join([char for char in search_base_pack1 if char.isdigit()])
#     search_uom1 = ''.join([char for char in search_base_pack1 if char.isalpha()])
#     search_itemdesc1 = str(transaction_file['ITEMDESC'][entry_index])

#     print(f"""\n|INFO| Search values (from transaction file) for entry '{entry_index}':
#           ITEMCODE: {search_itemcode1}
#           CATCODE: {search_catcode1}
#           COMPANY: {search_company1}
#           BRAND: {search_brand1}
#           PACKTYPE: {search_packtype1}
#           BASE PACK: {search_base_pack1}
#           QTY: {search_qty1}
#           UOM: {search_uom1}
#           ITEMDESC: {search_itemdesc1}""")

#     filtered_df = master_file.copy()
#     last_successful_df = pd.DataFrame()

#     # Pass 1: catcode (exact match)
#     if column_exists(filtered_df, 'catcode'):
#         filtered_df = filtered_df[filtered_df['catcode'].astype(str).str.strip() == search_catcode1.strip()]
#         if not filtered_df.empty:
#             last_successful_df = filtered_df.copy()
#         print(f"|INFO| After catcode filter: {len(filtered_df)} rows remain")
#         save_pass_df(filtered_df, "catcode", entry_index)

#     # Pass 2: company (partial and case-insensitive match)
#     if column_exists(filtered_df, 'company'):
#         filtered_df = filtered_df[filtered_df['company'].astype(str).str.contains(search_company1, case=False, na=False)]
#         if not filtered_df.empty:
#             last_successful_df = filtered_df.copy()
#         print(f"|INFO| After company filter: {len(filtered_df)} rows remain")
#         save_pass_df(filtered_df, "company", entry_index)

#     # Pass 3: brand (partial and case-insensitive match)
#     if column_exists(filtered_df, 'brand'):
#         filtered_df = filtered_df[filtered_df['brand'].astype(str).str.contains(search_brand1, case=False, na=False)]
#         if not filtered_df.empty:
#             last_successful_df = filtered_df.copy()
#         print(f"|INFO| After brand filter: {len(filtered_df)} rows remain")
#         save_pass_df(filtered_df, "brand", entry_index)

#     # Pass 4: packtype (exact and case-insensitive match)
#     if column_exists(filtered_df, 'packtype'):
#         filtered_df = filtered_df[filtered_df['packtype'].astype(str).str.lower().str.strip() == search_packtype1.lower().strip()]
#         if not filtered_df.empty:
#             last_successful_df = filtered_df.copy()
#         print(f"|INFO| After packtype filter: {len(filtered_df)} rows remain")
#         save_pass_df(filtered_df, "packtype", entry_index)

#     # Pass 5: qty + uom (exact numerical match for qty, exact case-insensitive match for uom)
#     if column_exists(filtered_df, 'qty') and column_exists(filtered_df, 'uom'):
#         try:
#             search_qty_num = pd.to_numeric(search_qty1)
#             qty_mask = filtered_df['qty'].astype(float) == search_qty_num
#             uom_mask = filtered_df['uom'].astype(str).str.lower().str.strip() == search_uom1.lower().strip()
#             filtered_df = filtered_df[qty_mask & uom_mask]
#             if not filtered_df.empty:
#                 last_successful_df = filtered_df.copy()
#         except ValueError:
#             print(f"|WARNING| Could not convert search_qty1 ('{search_qty1}') to a number. Skipping qty+uom pass.")
        
#         print(f"|INFO| After qty+uom filter: {len(filtered_df)} rows remain")
#         save_pass_df(filtered_df, "qty_uom", entry_index)


#     # Final output generation for this entry
#     if not last_successful_df.empty:
#         m_itemcodes = last_successful_df['itemcode'].unique().tolist()
#         m_itemcodes_str = ' || '.join([str(item) for item in m_itemcodes])
        
#         return pd.DataFrame({
#             't_itemcode': [search_itemcode1],
#             'm_itemcode(s)': [m_itemcodes_str]
#         })
    
#     return pd.DataFrame({
#         't_itemcode': [search_itemcode1],
#         'm_itemcode(s)': [None]
#     })


def process_transaction_entry(entry_index):
    """Process one transaction entry with pass-based filtering."""

    # Transaction values
    search_itemcode1 = str(transaction_file['ITEMCODE'][entry_index])
    search_catcode1 = str(transaction_file['CATEGORY'][entry_index])
    search_company1 = str(transaction_file['MANUFACTURE'][entry_index])
    search_brand1 = str(transaction_file['BRAND'][entry_index])
    search_packtype1 = normalize_value(transaction_file['PACKTYPE'][entry_index], packtype_mapping)
    search_base_pack1 = str(transaction_file['PACKSIZE'][entry_index])
    search_qty1 = ''.join([c for c in search_base_pack1 if c.isdigit()])
    search_uom1 = ''.join([c for c in search_base_pack1 if c.isalpha()])
    search_uom1 = normalize_value(search_uom1, uom_mapping)
    search_itemdesc1 = str(transaction_file['ITEMDESC'][entry_index])

    print(f"""\n|INFO| Search values for entry '{entry_index}':
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

    # Normalize master values
    filtered_df['uom'] = filtered_df['uom'].apply(lambda x: normalize_value(x, uom_mapping))
    filtered_df['packtype'] = filtered_df['packtype'].apply(lambda x: normalize_value(x, packtype_mapping))

    # Pass 1: catcode
    if column_exists(filtered_df, 'catcode'):
        filtered_df = filtered_df[filtered_df['catcode'].astype(str).str.strip() == search_catcode1.strip()]
        if filtered_df.empty:
            return pd.DataFrame({'t_itemcode': [search_itemcode1], 'm_itemcode(s)': [None]})
        save_pass_df(filtered_df, "catcode", entry_index)

    # Pass 2: company (fuzzy, keep top 5)
    if column_exists(filtered_df, 'company'):
        filtered_df['company_score'] = filtered_df['company'].apply(
            lambda x: fuzz.partial_ratio(str(x).lower(), search_company1.lower())
        )
        filtered_df = filtered_df.sort_values('company_score', ascending=False).head(5)
        if filtered_df.empty:
            return pd.DataFrame({'t_itemcode': [search_itemcode1], 'm_itemcode(s)': [None]})
        save_pass_df(filtered_df, "company", entry_index)

    # Pass 3: brand (fuzzy, keep top 5)
    if column_exists(filtered_df, 'brand'):
        filtered_df['brand_score'] = filtered_df['brand'].apply(
            lambda x: fuzz.partial_ratio(str(x).lower(), search_brand1.lower())
        )
        filtered_df = filtered_df.sort_values('brand_score', ascending=False).head(5)
        if filtered_df.empty:
            return pd.DataFrame({'t_itemcode': [search_itemcode1], 'm_itemcode(s)': [None]})
        save_pass_df(filtered_df, "brand", entry_index)

    # Pass 4: qty+uom
    search_qty_base = convert_qty_uom(search_qty1, search_uom1)
    filtered_df['qty_base'] = filtered_df.apply(lambda r: convert_qty_uom(r['qty'], r['uom']), axis=1)
    filtered_df = filtered_df[filtered_df['qty_base'] == search_qty_base]
    if filtered_df.empty:
        return pd.DataFrame({'t_itemcode': [search_itemcode1], 'm_itemcode(s)': [None]})
    save_pass_df(filtered_df, "qty_uom", entry_index)

    # Pass 5: packtype
    if column_exists(filtered_df, 'packtype'):
        filtered_df = filtered_df[filtered_df['packtype'] == search_packtype1]
        if filtered_df.empty:
            return pd.DataFrame({'t_itemcode': [search_itemcode1], 'm_itemcode(s)': [None]})
        save_pass_df(filtered_df, "packtype", entry_index)

    # Final Top 3 by combined score
    filtered_df['final_score'] = (filtered_df['company_score'] + filtered_df['brand_score']) / 2
    filtered_df = filtered_df.sort_values('final_score', ascending=False).head(3)

    # Collect itemcodes and scores (aligned order after sorting)
    m_itemcodes = filtered_df['itemcode'].astype(str).tolist()
    m_scores = filtered_df['final_score'].round(2).astype(str).tolist()

    m_itemcodes_str = ' || '.join(m_itemcodes)
    m_scores_str = ' || '.join(m_scores)

    return pd.DataFrame({
        't_itemcode': [search_itemcode1],
        'm_itemcode(s)': [m_itemcodes_str],
        'final_score': [m_scores_str]
    })


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


if __name__ == "__main__":
    main()