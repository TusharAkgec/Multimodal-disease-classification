import os
import shutil
import pandas as pd
from pathlib import Path

def main():
    # Paths
    base_dir = Path("data")
    csv_path = base_dir / "nih_metadata_prepped.csv"
    images_dir = base_dir / "images"
    extra_dir = base_dir / "images_extra"

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Create target directory if it doesn't exist
    extra_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse CSV and extract set of all image filenames
    df = pd.read_csv(csv_path)
    
    filename_col = None
    for c in df.columns:
        if 'filename' in c.lower() or c.lower() == 'filename' or 'image_name' in c.lower() or 'image index' in c.lower():
            filename_col = c
            break
            
    if filename_col is None:
        filename_col = df.columns[0] # Fallback to first column
        
    print(f"Using column '{filename_col}' to extract filenames from CSV.")

    # Get set of all filenames
    # Strip any whitespace and make sure they're strings
    csv_filenames = set(df[filename_col].astype(str).str.strip())
    
    # 2. Iterate through files and move
    moved_count = 0
    total_found_in_folder = 0
    
    for file_path in images_dir.iterdir():
        if not file_path.is_file():
            continue
            
        total_found_in_folder += 1
        
        # Check by actual filename and filename without extension (just in case)
        if file_path.name not in csv_filenames and file_path.stem not in csv_filenames:
            # Move it to extra folder
            target_path = extra_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            moved_count += 1

    # 3. Output Summary
    final_count = sum(1 for p in images_dir.iterdir() if p.is_file())
    
    print("-" * 40)
    print("Cleanup Summary")
    print("-" * 40)
    print(f"Total unique images referenced in CSV: {len(csv_filenames)}")
    print(f"Total images initially in data/images: {total_found_in_folder}")
    print(f"Total images moved to images_extra:    {moved_count}")
    print(f"Final count of images in data/images:  {final_count}")

if __name__ == "__main__":
    main()
