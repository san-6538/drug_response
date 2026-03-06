def encode_interaction_features(df):
    print("Engineering interaction features...")
    df["gene_drug"] = df["Gene"].astype(str) + "*" + df["Drug"].astype(str)
    df["variant_drug"] = df["Variant"].astype(str) + "*" + df["Drug"].astype(str)
    df["gene_variant"] = df["Gene"].astype(str) + "_" + df["Variant"].astype(str)
    
    return df
