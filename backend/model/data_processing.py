import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
df = pd.read_csv("data/vienna_apartments.csv")
vienna_district_map_numeric = {
    '1010': '01',
    '1020': '02',
    '1030': '03',
    '1040': '04',
    '1050': '05',
    '1060': '06',
    '1070': '07',
    '1080': '08',
    '1090': '09',
    '1100': '10',
    '1110': '11',
    '1120': '12',
    '1130': '13',
    '1140': '14',
    '1150': '15',
    '1160': '16',
    '1170': '17',
    '1180': '18',
    '1190': '19',
    '1200': '20',
    '1210': '21',
    '1220': '22',
    '1230': '23'
}
df.dropna(subset=["price"], inplace=True)
df.dropna(subset=["area_sqm"], inplace=True)
df.dropna(subset=["rooms"], inplace=True)
df = df[df["price"] >= 400]
df = df[df["price"] <= 5000]
df["district"] = df["address"].apply(
            lambda x: vienna_district_map_numeric[str(re.search(r"\d{4}", x).group(0))] if re.search(r"\d{4}", x) else 0)
df = df.drop("address", axis=1)
plt.figure(figsize=(10, 6))
district_counts = df['district'].value_counts()
#sns.histplot(df["district"], bins=50, kde=True)
sns.barplot(x=district_counts.index, y=district_counts.values, palette="viridis")
plt.xlabel("district")
plt.ylabel("quantity")
#plt.show()

print(df.iloc[1])
