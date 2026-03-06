import pandas as pd

def merge_datasets(clinical_variants, clinical_annotations, variants, drugs, guidelines_df):
    print("Merging core datasets...")
    
    cv_cols = {"gene_symbol": "Gene", "variant_name": "Variant", "drug_name": "Drug", "phenotype": "Phenotype"}
    cv = clinical_variants.rename(columns=lambda x: cv_cols.get(x, x))
    
    if not clinical_annotations.empty:
        ca = clinical_annotations.rename(columns=lambda x: cv_cols.get(x, x))
        main_df = pd.merge(cv, ca, on=["Gene", "Variant", "Drug"], how="outer", suffixes=("", "_ca"))
        if 'Phenotype_ca' in main_df.columns:
            main_df['Phenotype'] = main_df['Phenotype'].fillna(main_df['Phenotype_ca'])
            main_df = main_df.drop(columns=['Phenotype_ca'])
    else:
        main_df = cv.copy()

    main_df = main_df.dropna(subset=["Gene", "Variant", "Drug", "Phenotype"])

    if not variants.empty:
        v = variants.rename(columns={"variant_name": "Variant", "impact": "Variant_impact"})
        main_df = pd.merge(main_df, v, on="Variant", how="left")
    
    if not drugs.empty:
        d = drugs.rename(columns={"drug_name": "Drug", "drug_class": "Drug Class"})
        main_df = pd.merge(main_df, d, on="Drug", how="left")

    if not guidelines_df.empty:
        g = guidelines_df.rename(columns={"gene": "Gene", "variant": "Variant", "drug": "Drug"})
        
        # Drop Phenotype from guidelines to avoid Phenotype_x / Phenotype_y
        g = g.drop(columns=["Phenotype"], errors="ignore")
        
        # If cv already provided an evidence_level in some form, rename it to avoid collision
        if "evidence_level" in main_df.columns:
            main_df = main_df.rename(columns={"evidence_level": "evidence_level_cv"})
        if "Evidence_Level" in main_df.columns:
            main_df = main_df.rename(columns={"Evidence_Level": "evidence_level_cv"})
            
        g = g.drop_duplicates(subset=["Gene", "Variant", "Drug"])
        main_df = pd.merge(main_df, g, on=["Gene", "Variant", "Drug"], how="left")
        
        # Merge the evidence levels if we had a cv one
        if "evidence_level_cv" in main_df.columns:
            main_df["evidence_level"] = main_df["evidence_level"].fillna(main_df["evidence_level_cv"])
            main_df = main_df.drop(columns=["evidence_level_cv"])
    else:
        main_df['evidence_level'] = None
        main_df['recommendation'] = None

    if "evidence_level" not in main_df.columns:
        main_df["evidence_level"] = None
        
    main_df['guideline_exists'] = main_df['evidence_level'].notnull().astype(int)
    
    def map_evidence(val):
        if pd.isnull(val): return 0
        v_str = str(val).upper()
        if 'A' in v_str: return 3
        if 'B' in v_str: return 2
        if 'C' in v_str: return 1
        return 0
    main_df['evidence_level_num'] = main_df['evidence_level'].apply(map_evidence)

    main_df['recommendation'] = main_df['recommendation'].fillna("None").astype(str)

    if 'Drug Class' not in main_df.columns:
        main_df['Drug Class'] = "Unknown"
        
    main_df['Drug Class'] = main_df['Drug Class'].fillna("Unknown")

    if 'Variant_impact' not in main_df.columns and 'variant_impact' in main_df.columns:
        main_df['Variant_impact'] = main_df['variant_impact']
    elif 'Variant_impact' not in main_df.columns:
        main_df['Variant_impact'] = "Unknown"
        
    main_df['Variant_impact'] = main_df['Variant_impact'].fillna("Unknown")

    return main_df
