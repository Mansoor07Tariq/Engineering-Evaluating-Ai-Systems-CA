import pandas as pd
from Config import Config  #This imports your configuration class with field names

#Function to load and prepare the input data
def get_input_data() -> pd.DataFrame:
    """
    This function reads two CSV files: AppGallery and Purchasing,
    renames their label columns to standard names, merges them into
    one DataFrame, and prepares them for model input.
    """

    #Load both datasets
    df1 = pd.read_csv("data/AppGallery.csv", skipinitialspace=True)
    df2 = pd.read_csv("data/Purchasing.csv", skipinitialspace=True)

    #Rename class columns to y1, y2, y3, y4
    rename_map = {
        'Type 1': 'y1',
        'Type 2': 'y2',
        'Type 3': 'y3',
        'Type 4': 'y4'
    }
    df1.rename(columns=rename_map, inplace=True)
    df2.rename(columns=rename_map, inplace=True)

    #Combine both datasets into a single DataFrame
    df = pd.concat([df1, df2], ignore_index=True)

    #rint column names (useful for debugging)
    print("Columns in DataFrame:", df.columns.tolist())

    #Rename label columns (y1–y4) to final target names
    df.rename(columns={
        'y1': 'Intent',
        'y2': 'Tone',
        'y3': 'Resolution',
        'y4': 'Other'
    }, inplace=True)

    #Ensure interaction and summary columns are strings
    if Config.INTERACTION_CONTENT in df.columns:
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)

    if Config.TICKET_SUMMARY in df.columns:
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    #Set the main label column 'y' based on the configured target
    if Config.CLASS_COL in df.columns:
        df["y"] = df[Config.CLASS_COL]
    else:
        raise KeyError(
            f"Classification column '{Config.CLASS_COL}' not found! "
            f"Available columns: {df.columns.tolist()}"
        )

    #Drop rows with missing or empty labels
    df = df.loc[(df["y"] != '') & (~df["y"].isna())]

    return df
