import pandas as pd

dataset = pd.read_csv("data/vienna_apartments.csv")

dataset["price_per_qm"] = dataset["price"] / dataset["area_sqm"]
dataset.to_csv("data/vienna_test.csv")