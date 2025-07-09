from pyinaturalist import *
import time
import pandas as pd

"""
    This script is responsible for iterating over species described in 'Polish-Mushroom-Dataset.xlsx' and getting observations of the species from INaturalist. The script receives 840 images URLs for each out of 300 species, which gives around 250 000 images. 
"""

dfs = pd.read_excel("Polish-Mushroom-Dataset.xlsx", "Shrooms")

with open("species-and-photos.csv", "a+") as file:
    for i1, row in dfs.iterrows():
        species = row["Species"]
        quantity = int(row["Number of observations on INaturalist"])
        if quantity < 840:
            continue

        for i2 in range(1, 6):
            per_page = 200
            if i2 == 5:
                per_page = 40
            
            observations = get_observations(
                taxon_name=species,
                photos=True,
                license=["CC0", "CC-BY", "CC-BY-NC", "CC-BY-SA", "CC-BY-NC-SA"],
                per_page=per_page,
                page=i2
            )
            for observation in observations["results"]:
                file.write(f"{species},{observation["id"]},{observation["photos"][0]["url"]}\n")
            time.sleep(2)
            print(f"{i1+1} {species}, page number #{i2} done")