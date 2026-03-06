def map_phenotype(p):
    p = str(p).lower()

    if "dose" in p or "dosage" in p:
        return "Dose_Adjustment"

    if "toxicity" in p or "adverse" in p or "risk" in p or "disease" in p or "injury" in p or "syndrome" in p or "hemorrhage" in p:
        return "Adverse_Reaction"

    if "metabolizer" in p or "metabolic" in p or "metabolism" in p or "pk" in p:
        return "Metabolism"

    if "efficacy" in p or "resistance" in p or "non-response" in p:
        return "Drug_Resistance"

    if "normal" in p or "unknown" in p:
        return "Normal_Response"

    # Some additional mappings based on common PharmGKB phenotypes
    if "response" in p or "outcome" in p:
        return "Normal_Response"

    return "Other"
