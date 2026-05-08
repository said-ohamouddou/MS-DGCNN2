import os
import pandas as pd
import random
from pathlib import Path

def sample_tree_files(base_directory, sample_ratio=0.2, output_csv="sampled_files.csv"):
    """
    Sample files from tree species directories and save to CSV.
    
    Args:
        base_directory (str): Path to directory containing the 7 tree species folders
        sample_ratio (float): Ratio of files to sample (0.2 = 20%)
        output_csv (str): Output CSV filename
    """
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # List to store sampled file information
    sampled_files = []
    
    # Get all directories in the base directory
    base_path = Path(base_directory)
    tree_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(tree_dirs)} directories")
    
    for tree_dir in tree_dirs:
        # Get the class name (directory name)
        class_name = tree_dir.name
        
        # Find all .pts, .txt, and .xyz files
        file_extensions = ['.pts', '.txt', '.xyz']
        all_files = []
        
        for ext in file_extensions:
            files = list(tree_dir.glob(f"*{ext}"))
            all_files.extend(files)
        
        print(f"Class '{class_name}': Found {len(all_files)} files")
        
        if len(all_files) == 0:
            print(f"Warning: No files found in {tree_dir}")
            continue
        
        # Calculate number of files to sample
        num_to_sample = max(1, int(len(all_files) * sample_ratio))
        
        # Randomly sample files
        sampled = random.sample(all_files, num_to_sample)
        
        # Add to results
        for file_path in sampled:
            sampled_files.append({
                'path': str(file_path),
                'class': class_name
            })
        
        print(f"Sampled {num_to_sample} files from class '{class_name}'")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sampled_files)
    
    # Sort by class for better organization
    df = df.sort_values('class').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nSampling complete!")
    print(f"Total files sampled: {len(df)}")
    print(f"Results saved to: {output_csv}")
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = df['class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} files")
    
    return df

def main():
    # Configuration
    BASE_DIRECTORY = "."  # Change this to your base directory path
    SAMPLE_RATIO = 0.2    # 20% sampling
    OUTPUT_CSV = "sampled_tree_files.csv"
    
    # Run the sampling
    try:
        result_df = sample_tree_files(
            base_directory=BASE_DIRECTORY,
            sample_ratio=SAMPLE_RATIO,
            output_csv=OUTPUT_CSV
        )
        
        # Display first few rows
        print(f"\nFirst 10 rows of the sampled data:")
        print(result_df.head(10))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the directory path is correct and contains subdirectories with tree files.")

if __name__ == "__main__":
    main()
