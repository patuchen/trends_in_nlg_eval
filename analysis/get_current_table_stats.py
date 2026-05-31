import pandas as pd
import os

analysis_dir = os.path.dirname(os.path.abspath(__file__))
tables_dir = os.path.join(analysis_dir, "tables")

for fn in sorted(os.listdir(tables_dir)):
    if fn.endswith(".csv"):
        print(f"\n==================== {fn} ====================")
        df = pd.read_csv(os.path.join(tables_dir, fn))
        print("Columns:", list(df.columns))
        print("Shape:", df.shape)
        print("Head:")
        print(df.head(3).to_string())
