from src.features.phenotype_mapping import map_phenotype

def preprocess_target(df):
    print("Preprocessing target (applying phenotype mapping)...")
    
    # 1. Map fine-grained phenotypes to broad clinical categories
    df["Phenotype"] = df["Phenotype"].apply(map_phenotype)
    
    # 2. Optionally, still group any that are extremely rare
    counts = df["Phenotype"].value_counts()
    valid_phenotypes = counts[counts >= 5].index
    
    df.loc[~df["Phenotype"].isin(valid_phenotypes), "Phenotype"] = "Other"
    
    return df
