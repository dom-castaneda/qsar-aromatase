"""
Fetch bioactivity data (Ki, IC50, pIC50, pChEMBL) for Aromatase (CHEMBL1978)
from the ChEMBL REST API.
"""

import requests
import csv
import time

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGET_CHEMBL_ID = "CHEMBL1978"
STANDARD_TYPES = "Ki,IC50,pIC50"
LIMIT = 1000
OUTPUT_FILE = "../data/raw/aromatase_bioactivity.csv"

COLUMNS = [
    "molecule_chembl_id",
    "molecule_pref_name",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "target_chembl_id",
    "target_pref_name",
    "target_organism",
    "document_chembl_id",
    "document_journal",
    "document_year",
    "activity_id",
]


def fetch_all_activities():
    all_activities = []
    offset = 0

    while True:
        params = {
            "target_chembl_id": TARGET_CHEMBL_ID,
            "standard_type__in": STANDARD_TYPES,
            "limit": LIMIT,
            "offset": offset,
        }
        print(f"Fetching offset {offset} ...")
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        activities = data.get("activities", [])
        all_activities.extend(activities)

        total = data["page_meta"]["total_count"]
        print(f"  Got {len(activities)} records (total so far: {len(all_activities)}/{total})")

        if data["page_meta"]["next"] is None:
            break

        offset += LIMIT
        time.sleep(0.5)

    return all_activities


def write_csv(activities, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for act in activities:
            writer.writerow(act)


def main():
    print(f"Fetching bioactivity data for Aromatase ({TARGET_CHEMBL_ID}) ...")
    print(f"Activity types: Ki, IC50, pIC50 (pChEMBL values included where available)\n")

    activities = fetch_all_activities()
    write_csv(activities, OUTPUT_FILE)

    ki_count = sum(1 for a in activities if a["standard_type"] == "Ki")
    ic50_count = sum(1 for a in activities if a["standard_type"] == "IC50")
    pic50_count = sum(1 for a in activities if a["standard_type"] == "pIC50")
    pchembl_count = sum(1 for a in activities if a.get("pchembl_value") is not None)

    print(f"\n{'='*50}")
    print(f"Total records:    {len(activities)}")
    print(f"  Ki records:     {ki_count}")
    print(f"  IC50 records:   {ic50_count}")
    print(f"  pIC50 records:  {pic50_count}")
    print(f"  With pChEMBL:   {pchembl_count}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
