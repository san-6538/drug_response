import pandas as pd
import json
import glob
import os

def load_datasets(data_dir="data"):
    print("Loading CSV datasets...")
    
    def safe_read(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f"Warning: {path} not found.")
            return pd.DataFrame()
            
    clinical_variants = safe_read(os.path.join(data_dir, "clinical_variants_cleaned.csv"))
    clinical_annotations = safe_read(os.path.join(data_dir, "clinical_annotations_cleaned.csv"))
    genes = safe_read(os.path.join(data_dir, "genes_cleaned.csv"))
    variants = safe_read(os.path.join(data_dir, "variants_cleaned.csv"))
    drugs = safe_read(os.path.join(data_dir, "drugs_cleaned.csv"))
    relationships = safe_read(os.path.join(data_dir, "relationships_cleaned.csv"))

    print("Loading guideline JSONs...")
    guideline_files = glob.glob(os.path.join(data_dir, "guidelines", "*.json"))
    guidelines_list = []
    for f in guideline_files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        guidelines_list.append({
                            "Gene": item.get("gene", ""),
                            "Variant": item.get("variant", ""),
                            "Drug": item.get("drug", ""),
                            "Phenotype": item.get("phenotype", ""),
                            "recommendation": item.get("recommendation", ""),
                            "evidence_level": item.get("evidence_level", "")
                        })
                else:
                    guidelines_list.append({
                        "Gene": data.get("gene", ""),
                        "Variant": data.get("variant", ""),
                        "Drug": data.get("drug", ""),
                        "Phenotype": data.get("phenotype", ""),
                        "recommendation": data.get("recommendation", ""),
                        "evidence_level": data.get("evidence_level", "")
                    })
            except json.JSONDecodeError:
                pass

    guidelines_df = pd.DataFrame(guidelines_list)
    print(f"Loaded {len(guidelines_df)} guideline records.")

    return clinical_variants, clinical_annotations, genes, variants, drugs, relationships, guidelines_df
