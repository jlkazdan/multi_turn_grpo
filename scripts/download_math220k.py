#!/usr/bin/env python3
"""
Download OpenR1-Math-220k dataset from Hugging Face Hub
"""

import os
from datasets import load_dataset
from pathlib import Path

def download_openr1_math_220k():
    """Download OpenR1-Math-220k dataset with all configurations"""
    
    # Set cache directory to current data folder
    cache_dir = Path(__file__).parent / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    
    print("=== Downloading OpenR1-Math-220k Dataset ===")
    print(f"Cache directory: {cache_dir}")
    
    # Download all three configurations
    configs = ["default", "all", "extended"]
    
    for config in configs:
        print(f"\n📥 Downloading configuration: {config}")
        try:
            ds = load_dataset(
                "open-r1/OpenR1-Math-220k", 
                config,
                cache_dir=str(cache_dir)
            )
            
            # Save as parquet files
            output_file = Path(__file__).parent / f"openr1-math-220k-{config}.parquet"
            
            # Combine all splits if multiple exist
            if len(ds) == 1:
                # Single split
                split_name = list(ds.keys())[0]
                df = ds[split_name].to_pandas()
            else:
                # Multiple splits - combine them
                import pandas as pd
                dfs = []
                for split_name, split_data in ds.items():
                    split_df = split_data.to_pandas()
                    split_df['split'] = split_name
                    dfs.append(split_df)
                df = pd.concat(dfs, ignore_index=True)
            
            # Save to parquet
            df.to_parquet(output_file, index=False)
            
            print(f"✅ Saved {len(df)} samples to {output_file}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show sample
            if len(df) > 0:
                print(f"   Sample keys: {list(df.iloc[0].keys())}")
                
        except Exception as e:
            print(f"❌ Failed to download {config}: {e}")
    
    print(f"\n✅ Download complete! Files saved in {Path(__file__).parent}")

if __name__ == "__main__":
    download_openr1_math_220k()