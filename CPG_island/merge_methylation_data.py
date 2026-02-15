#!/usr/bin/env python3
"""
Merge methylation beta matrix with CpG annotations and patient disease metadata.

This script:
1. Loads patient metadata and creates sample-to-disease mapping
2. Filters CpG annotations to autosomal chromosomes only (chr1-22)
3. Processes the large 16GB beta matrix in memory-safe chunks
4. Merges CpG annotations with beta values
5. Saves as compressed Parquet format

Author: Auto-generated
Date: 2025-11-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc

# File paths
PATIENT_METADATA_FILE = 'patient_metadata.csv'
CPG_INFO_FILE = 'cpg_info.csv'
BETA_MATRIX_FILE = 'GSE145361_Vallerga2020_NCOMMS_AvgBeta_Matrix-file(1).txt'
OUTPUT_DIR = Path('merged_methylation_data')  # Output directory for chunk files
OUTPUT_METADATA = 'sample_disease_mapping.csv'
OUTPUT_SUMMARY = 'merge_summary.txt'

# Configuration
CHUNK_SIZE = 10000  # Process 10K CpG sites at a time (smaller for memory efficiency)
AUTOSOMAL_CHROMS = [str(i) for i in range(1, 23)]  # chr1-22


def print_progress(message):
    """Print timestamped progress message"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_patient_metadata():
    """
    Load patient metadata and create sample-to-disease mapping.

    Returns:
        pd.DataFrame: Cleaned patient metadata
    """
    print_progress("Loading patient metadata...")

    df = pd.read_csv(PATIENT_METADATA_FILE)

    # Clean up extra quotes in Patient_ID
    df['Patient_ID'] = df['Patient_ID'].str.strip('"')

    print_progress(f"  Loaded {len(df)} samples")
    print_progress(f"  Disease distribution:")
    print(df['Disease_State'].value_counts().to_string())

    return df


def load_and_filter_cpg_info():
    """
    Load CpG annotation info and filter to autosomal chromosomes only.

    Returns:
        pd.DataFrame: Filtered CpG annotations (chr1-22 only)
    """
    print_progress("Loading CpG annotation info...")

    df = pd.read_csv(CPG_INFO_FILE)
    total_cpgs = len(df)

    print_progress(f"  Total CpG sites: {total_cpgs:,}")

    # Filter to autosomal chromosomes only
    df['CHR'] = df['CHR'].astype(str)
    df_autosomal = df[df['CHR'].isin(AUTOSOMAL_CHROMS)].copy()

    autosomal_cpgs = len(df_autosomal)
    removed_cpgs = total_cpgs - autosomal_cpgs

    print_progress(f"  Autosomal CpG sites (chr1-22): {autosomal_cpgs:,}")
    print_progress(f"  Removed (sex chromosomes + other): {removed_cpgs:,}")

    # Keep only relevant columns
    cols_to_keep = ['IlmnID', 'CHR', 'MAPINFO', 'Strand',
                    'Infinium_Design_Type', 'Genome_Build']
    df_autosomal = df_autosomal[cols_to_keep]

    return df_autosomal


def process_and_merge_chunks(cpg_autosomal_ids, cpg_info):
    """
    Process the large beta matrix file in chunks and merge with annotations.
    Writes each chunk to separate parquet file to avoid memory issues.

    Args:
        cpg_autosomal_ids: Set of autosomal CpG IDs to keep
        cpg_info: DataFrame with CpG annotations

    Returns:
        tuple: (total_rows_processed, total_rows_kept, sample_columns, num_chunks)
    """
    print_progress("Processing beta matrix in chunks and writing output...")
    print_progress(f"  File size: ~16GB")
    print_progress(f"  Chunk size: {CHUNK_SIZE:,} rows")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_rows_kept = 0
    total_rows_processed = 0
    chunk_num = 0
    chunks_written = 0
    sample_columns = None

    # Use tab separator and read in chunks
    chunks = pd.read_csv(
        BETA_MATRIX_FILE,
        sep='\t',
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk in chunks:
        chunk_num += 1
        chunk_size = len(chunk)
        total_rows_processed += chunk_size

        # Filter to autosomal CpGs only
        chunk_filtered = chunk[chunk['IID'].isin(cpg_autosomal_ids)]
        rows_kept = len(chunk_filtered)
        total_rows_kept += rows_kept

        if rows_kept > 0:
            # Merge with CpG info
            chunk_merged = chunk_filtered.merge(
                cpg_info,
                left_on='IID',
                right_on='IlmnID',
                how='left'
            )

            # Drop duplicate IID column
            chunk_merged = chunk_merged.drop(columns=['IID'])

            # Reorder columns: annotations first, then sample beta values
            annotation_cols = ['IlmnID', 'CHR', 'MAPINFO', 'Strand',
                               'Infinium_Design_Type', 'Genome_Build']

            if sample_columns is None:
                sample_columns = [col for col in chunk_merged.columns if col not in annotation_cols]

            chunk_merged = chunk_merged[annotation_cols + sample_columns]

            # Convert MAPINFO to int
            chunk_merged['MAPINFO'] = chunk_merged['MAPINFO'].astype('Int64')

            # Write chunk to separate parquet file
            chunk_file = OUTPUT_DIR / f"chunk_{chunks_written:04d}.parquet"
            chunk_merged.to_parquet(chunk_file, compression='snappy', index=False)
            chunks_written += 1

            print_progress(f"  Chunk {chunk_num}: Processed {chunk_size:,} rows, "
                          f"kept {rows_kept:,} autosomal CpGs, wrote to {chunk_file.name} "
                          f"(Total kept: {total_rows_kept:,})")

        # Force garbage collection to free memory
        del chunk
        if 'chunk_filtered' in locals():
            del chunk_filtered
        if 'chunk_merged' in locals():
            del chunk_merged
        gc.collect()

    print_progress(f"  Total rows processed: {total_rows_processed:,}")
    print_progress(f"  Total autosomal CpGs kept: {total_rows_kept:,}")
    print_progress(f"  Chunks written: {chunks_written}")

    return total_rows_processed, total_rows_kept, sample_columns, chunks_written


def save_outputs(total_rows, sample_columns, num_chunks, patient_metadata, cpg_info):
    """
    Save metadata and generate summary.

    Args:
        total_rows: Total number of CpG sites kept
        sample_columns: List of sample column names
        num_chunks: Number of chunk files written
        patient_metadata: Patient metadata with disease info
        cpg_info: CpG annotation info
    """
    print_progress("Saving metadata and summary...")

    # Calculate total directory size
    total_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.parquet')) / (1024 * 1024)

    # Save sample-to-disease mapping
    print_progress(f"  Saving metadata to {OUTPUT_METADATA}...")
    patient_metadata.to_csv(OUTPUT_METADATA, index=False)

    # Generate summary
    print_progress(f"  Generating summary to {OUTPUT_SUMMARY}...")

    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("METHYLATION DATA MERGE SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("INPUT FILES:\n")
        f.write(f"  - Patient metadata: {PATIENT_METADATA_FILE}\n")
        f.write(f"  - CpG annotations: {CPG_INFO_FILE}\n")
        f.write(f"  - Beta matrix: {BETA_MATRIX_FILE}\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  - Merged data directory: {OUTPUT_DIR}/\n")
        f.write(f"  - Number of chunk files: {num_chunks}\n")
        f.write(f"  - Total size: {total_size_mb:.1f} MB\n")
        f.write(f"  - Sample metadata: {OUTPUT_METADATA}\n\n")

        f.write("DATASET DIMENSIONS:\n")
        f.write(f"  - CpG sites (rows): {total_rows:,}\n")
        f.write(f"  - Samples (columns): {len(sample_columns)}\n")
        f.write(f"  - Total data points: {(total_rows * len(sample_columns)):,}\n\n")

        f.write("CHROMOSOMES INCLUDED:\n")
        f.write(f"  - Autosomal only: chr1-22\n")
        f.write(f"  - Sex chromosomes (X, Y) excluded\n\n")

        f.write("CpG DISTRIBUTION BY CHROMOSOME:\n")
        chrom_counts = cpg_info['CHR'].value_counts().sort_index()
        for chrom, count in chrom_counts.items():
            f.write(f"  chr{chrom}: {count:,} CpG sites\n")

        f.write("\nDISEASE DISTRIBUTION:\n")
        disease_counts = patient_metadata['Disease_State'].value_counts()
        for disease, count in disease_counts.items():
            f.write(f"  {disease}: {count} samples\n")

        f.write("\n" + "=" * 60 + "\n")

    print_progress("  Summary saved!")


def main():
    """Main execution function"""
    start_time = time.time()

    print_progress("=" * 60)
    print_progress("METHYLATION DATA MERGING PIPELINE")
    print_progress("=" * 60)

    # Step 1: Load patient metadata
    patient_metadata = load_patient_metadata()

    # Step 2: Load and filter CpG annotations
    cpg_info = load_and_filter_cpg_info()
    autosomal_cpg_ids = set(cpg_info['IlmnID'])

    # Step 3: Process beta matrix in chunks and merge with annotations
    total_processed, total_kept, sample_cols, num_chunks = process_and_merge_chunks(
        autosomal_cpg_ids, cpg_info
    )

    # Step 4: Save metadata and summary
    save_outputs(total_kept, sample_cols, num_chunks, patient_metadata, cpg_info)

    # Final summary
    elapsed_time = time.time() - start_time
    print_progress("=" * 60)
    print_progress(f"PIPELINE COMPLETE in {elapsed_time:.1f} seconds "
                  f"({elapsed_time/60:.1f} minutes)")
    print_progress("=" * 60)

    print("\nOutput files created:")
    print(f"  1. {OUTPUT_DIR}/ - Merged methylation data ({num_chunks} Parquet chunk files)")
    print(f"  2. {OUTPUT_METADATA} - Sample-to-disease mapping (CSV)")
    print(f"  3. {OUTPUT_SUMMARY} - Summary statistics (TXT)")
    print(f"\nNOTE: Data is split across {num_chunks} chunk files to avoid memory issues.")
    print(f"      Each chunk file can be loaded separately or concatenated later if you have enough RAM.")


if __name__ == "__main__":
    main()
