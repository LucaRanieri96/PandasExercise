import pandas as pd

file_path = "../0.datasets/mushrooms.csv"

mush_df = pd.read_csv(file_path)

print(mush_df.info())
