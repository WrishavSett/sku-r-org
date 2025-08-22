import os
import pandas as pd


# MASTER FILE loading and preview
try:
    master_file = pd.read_excel("dataset/NP_ItemMaster_Detailed_2025-07.xlsx")
except FileNotFoundError:
    print("\n|ERROR| MASTER FILE not found")
    exit()

master_file_columns = master_file.columns.tolist()
print("\n|INFO| MASTER FILE columns:\n", master_file_columns)
print("|INFO| MASTER FILE head:\n", master_file.head())
print("|INFO| MASTER FILE unique packtype:\n", master_file['packtype'].unique())
print("|INFO| MASTER FILE unique uom:\n", master_file['uom'].unique())


# TRANSACTION FILE loading and preview
try:
    transaction_file = pd.read_excel("dataset/NP_NI_Cross-Re_2024-12.xlsx", sheet_name="aug-24")
except FileNotFoundError:
    print("\n|ERROR| TRANSACTION FILE not found")
    exit()

transaction_file_columns = transaction_file.columns.tolist()
print("\n|INFO| TRANSACTION FILE columns:\n", transaction_file_columns)
print("|INFO| TRANSACTION FILE head:\n", transaction_file.head())

packsize = []
uom = []

for index, row in transaction_file.iterrows():
    base_pack = row['PACKSIZE']
    packsize.append(''.join([char for char in base_pack if char.isdigit()]))
    uom.append(''.join([char for char in base_pack if not char.isdigit()]))

print("|INFO| TRANSACTION FILE unique packtype:\n", transaction_file['PACKTYPE'].unique())
print("|INFO| TRANSACTION FILE unique uom:\n", set(uom))