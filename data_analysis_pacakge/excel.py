import pandas as pd
import glob

def get_all_excel(folder_path: str) -> dict:
    all_files = glob.glob(f"{folder_path}/*.xlsx") + glob.glob(f"{folder_path}/*.xls") + glob.glob(f"{folder_path}/*.csv")

    df_dict = {}
    for file in all_files:
        df_name = file.split("/")[-1].split(".")[0]  
        if file.endswith(('.xlsx', '.xls')):
            df_dict[df_name] = pd.read_excel(file, sheet_name=None)
        elif file.endswith('.csv'):
            df_dict[df_name] = pd.read_csv(file)
    return df_dict