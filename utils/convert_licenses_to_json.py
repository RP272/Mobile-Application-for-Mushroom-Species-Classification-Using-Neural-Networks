import pandas as pd
import json

json_data = []

dfs = pd.read_excel("licenses.xlsx", na_filter=False)

for index, row in dfs.iterrows():

    json_data.append({
        "resourceName": row["resource_name"] if len(row["resource_name"]) > 0 else "",
        "resourceURL": row["resource_url"] if len(row["resource_url"]) > 0 else "",
        "resourceLicense": row["resource_license"] if len(row["resource_license"]) > 0 else "",
        "licenseURL": row["license_url"] if len(row["license_url"]) > 0 else "",
        "resourceSiteURL": row["resource_site_url"] if len(row["resource_site_url"]) > 0 else "",
        "source1URL": row["source1"] if len(row["source1"]) > 0 else "",
        "source2URL": row["source2"] if len(row["source2"]) > 0 else "",
        "source3URL": row["source3"] if len(row["source3"]) > 0 else "",
        "source4URL": row["source4"] if len(row["source4"]) > 0 else "",
        "author": row["resource_author"] if len(row["resource_author"]) > 0 else "",
    })

with open("licenses.json", "w") as f:
    json.dump(json_data, f)
