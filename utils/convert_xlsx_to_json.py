import pandas as pd
import json
import math

json_data = []

dfs = pd.read_excel("species_data.xlsx", na_filter=False)

for index, row in dfs.iterrows():

    json_data.append({
        "latinName": row["latin_name"] if len(row["latin_name"]) > 0 else "",
        "edibility": row["edibility"] if len(row["edibility"]) > 0 else "inedible",
        "image": row["image_name"] if len(row["image_name"]) >= 0 else "cnv1_19",
        "description": row["description"] if len(row["description"]) > 0 else "description"
    })

with open("mushrooms.json", "w") as f:
    json.dump(json_data, f)
