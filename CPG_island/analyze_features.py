#!/usr/bin/env python3
"""
Feature Analysis Script

Analyzes the selected CpG sites to identify the most important biomarkers
for Parkinson's disease classification.

Usage:
    python3 analyze_features.py
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("CpG BIOMARKER ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n1. Loading feature selection data...")
    stats = pd.read_csv('ml_analysis/feature_selection/selection_statistics.csv')
    importance = pd.read_csv('ml_analysis/feature_selection/feature_importance.csv')
    cpg_info = pd.read_csv('cpg_info.csv')

    print(f"   ✓ Loaded statistics for {len(stats):,} CpG sites")
    print(f"   ✓ Loaded importance scores from {len(importance['Model'].unique())} models")

    # Get Random Forest importance (best model)
    rf_imp = importance[importance['Model'] == 'Random Forest'].copy()
    print(f"   ✓ Using Random Forest feature importance")

    # Merge everything
    print("\n2. Merging all data sources...")
    full_data = stats.merge(rf_imp[['CpG_ID', 'Importance']], on='CpG_ID', how='left')
    full_data = full_data.merge(
        cpg_info[['IlmnID', 'CHR', 'MAPINFO', 'Strand']],
        left_on='CpG_ID',
        right_on='IlmnID',
        how='left'
    )

    # Clean up
    full_data = full_data.dropna(subset=['Importance'])  # Only selected features
    full_data['CHR'] = full_data['CHR'].astype(str)

    print(f"   ✓ Merged data for {len(full_data):,} selected CpG sites")

    # Define tiers based on combined criteria
    print("\n3. Categorizing CpG sites into tiers...")

    full_data['Tier'] = 'Medium'  # Default

    # High tier: significant p-value AND above-median importance
    high_mask = (full_data['p_value'] < 0.01) & \
                (full_data['Importance'] > full_data['Importance'].median())
    full_data.loc[high_mask, 'Tier'] = 'High'

    # Priority tier: very significant p-value AND top 25% importance
    priority_mask = (full_data['p_value'] < 0.001) & \
                    (full_data['Importance'] > full_data['Importance'].quantile(0.75))
    full_data.loc[priority_mask, 'Tier'] = 'Priority'

    # Low tier: less significant
    low_mask = full_data['p_value'] >= 0.01
    full_data.loc[low_mask, 'Tier'] = 'Low'

    # Summary by tier
    print("\n" + "=" * 80)
    print("FEATURE TIER SUMMARY")
    print("=" * 80)
    tier_counts = full_data['Tier'].value_counts()
    tier_order = ['Priority', 'High', 'Medium', 'Low']

    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(full_data) * 100
            print(f"  {tier:<10} {count:>4} CpG sites ({pct:>5.1f}%)")

    # Top priority biomarkers
    print("\n" + "=" * 80)
    print("TOP 20 PRIORITY BIOMARKERS")
    print("=" * 80)

    priority = full_data[full_data['Tier'] == 'Priority'].copy()
    priority = priority.sort_values('Importance', ascending=False)

    if len(priority) > 0:
        print(f"\n{len(priority)} priority biomarkers identified")
        print("\nTop 20 by importance:")
        print("-" * 80)
        print(f"{'Rank':<6} {'CpG ID':<15} {'Chr':<5} {'Position':<12} {'p-value':<12} {'Importance':<12}")
        print("-" * 80)

        for i, (idx, row) in enumerate(priority.head(20).iterrows(), 1):
            cpg_id = row['CpG_ID']
            chrom = f"chr{row['CHR']}" if pd.notna(row['CHR']) else "N/A"
            pos = f"{int(row['MAPINFO']):,}" if pd.notna(row['MAPINFO']) else "N/A"
            pval = f"{row['p_value']:.2e}"
            imp = f"{row['Importance']:.4f}"

            print(f"{i:<6} {cpg_id:<15} {chrom:<5} {pos:<12} {pval:<12} {imp:<12}")
    else:
        print("\nNo CpG sites met the priority criteria.")
        print("Showing top 20 by importance instead:")
        print("-" * 80)
        print(f"{'Rank':<6} {'CpG ID':<15} {'Chr':<5} {'Position':<12} {'p-value':<12} {'Importance':<12}")
        print("-" * 80)

        top20 = full_data.nlargest(20, 'Importance')
        for i, (idx, row) in enumerate(top20.iterrows(), 1):
            cpg_id = row['CpG_ID']
            chrom = f"chr{row['CHR']}" if pd.notna(row['CHR']) else "N/A"
            pos = f"{int(row['MAPINFO']):,}" if pd.notna(row['MAPINFO']) else "N/A"
            pval = f"{row['p_value']:.2e}"
            imp = f"{row['Importance']:.4f}"

            print(f"{i:<6} {cpg_id:<15} {chrom:<5} {pos:<12} {pval:<12} {imp:<12}")

    # Distribution by chromosome
    print("\n" + "=" * 80)
    print("DISTRIBUTION BY CHROMOSOME")
    print("=" * 80)

    chr_dist = full_data[full_data['CHR'].str.isnumeric()].copy()
    chr_dist['CHR_num'] = chr_dist['CHR'].astype(int)
    chr_counts = chr_dist.groupby('CHR_num').size().sort_index()

    print("\nNumber of selected CpG sites per chromosome:")
    print("-" * 80)
    for chrom, count in chr_counts.items():
        pct = count / len(full_data) * 100
        bar = "█" * int(pct / 2)  # Visual bar
        print(f"chr{chrom:>2}  {count:>4} CpG sites ({pct:>5.1f}%)  {bar}")

    # Statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    print(f"\np-value distribution:")
    print(f"  p < 0.001:  {(full_data['p_value'] < 0.001).sum():>5} CpG sites ({(full_data['p_value'] < 0.001).sum()/len(full_data)*100:>5.1f}%)")
    print(f"  p < 0.01:   {(full_data['p_value'] < 0.01).sum():>5} CpG sites ({(full_data['p_value'] < 0.01).sum()/len(full_data)*100:>5.1f}%)")
    print(f"  p < 0.05:   {(full_data['p_value'] < 0.05).sum():>5} CpG sites ({(full_data['p_value'] < 0.05).sum()/len(full_data)*100:>5.1f}%)")

    print(f"\nImportance scores:")
    print(f"  Mean:       {full_data['Importance'].mean():.6f}")
    print(f"  Median:     {full_data['Importance'].median():.6f}")
    print(f"  Max:        {full_data['Importance'].max():.6f}")
    print(f"  Min:        {full_data['Importance'].min():.6f}")

    print(f"\nF-scores:")
    print(f"  Mean:       {full_data['F_score'].mean():.2f}")
    print(f"  Median:     {full_data['F_score'].median():.2f}")
    print(f"  Max:        {full_data['F_score'].max():.2f}")

    # Save priority biomarkers
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_file = 'priority_biomarkers.csv'
    if len(priority) > 0:
        priority_export = priority[['CpG_ID', 'CHR', 'MAPINFO', 'Strand',
                                    'p_value', 'F_score', 'Importance', 'Tier']]
        priority_export.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(priority)} priority biomarkers to: {output_file}")
    else:
        top_export = full_data.nlargest(100, 'Importance')[['CpG_ID', 'CHR', 'MAPINFO', 'Strand',
                                                             'p_value', 'F_score', 'Importance', 'Tier']]
        top_export.to_csv(output_file, index=False)
        print(f"\n✓ Saved top 100 biomarkers to: {output_file}")

    # Save full annotated data
    full_output = 'all_features_annotated.csv'
    full_data.to_csv(full_output, index=False)
    print(f"✓ Saved complete annotated data to: {full_output}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review priority_biomarkers.csv for top candidates")
    print("  2. Map CpG sites to nearby genes using genomic databases")
    print("  3. Investigate biological significance of top chromosomes")
    print("  4. Consider pathway enrichment analysis")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
