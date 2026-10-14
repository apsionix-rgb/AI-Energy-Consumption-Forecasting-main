import pandas as pd
import os

def load_all_data(folder_path):
    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)

            df = pd.read_csv(file_path)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Detect datetime column
            datetime_col = None
            for col in df.columns:
                if "datetime" in col.lower():
                    datetime_col = col
                    break

            if datetime_col is None:
                continue

            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)

            # Rename energy column dynamically
            for col in df.columns:
                df.rename(columns={col: 'Energy'}, inplace=True)
                break

            # Add region name
            df['region'] = file.replace(".csv", "")

            all_data.append(df)

    combined = pd.concat(all_data)

    return combined