import pickle
import os
import json

def depickle_file(file_path):
    """
    Load and display contents of a pickled file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        The unpickled data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Successfully loaded: {file_path}")
        print(f"Data type: {type(data)}")
        
        # Display basic info about the data
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys:")
            for key in list(data.keys())[:10]:  # Show first 10 keys
                print(f"  - {key}: {type(data[key])}")
            if len(data) > 10:
                print(f"  ... and {len(data) - 10} more keys")
        
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            if data:
                print(f"First item type: {type(data[0])}")
        
        elif isinstance(data, tuple):
            print(f"Tuple with {len(data)} items")
        
        else:
            print(f"Data: {data}")
        
        return data
        
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        return None

def save_to_text_file(data, output_path, description=""):
    """
    Save data to a text file in a readable format.
    
    Args:
        data: Data to save
        output_path: Path to save the text file
        description: Optional description to add at the top
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if description:
                f.write(f"# {description}\n")
                f.write("=" * 50 + "\n\n")
            
            if isinstance(data, dict):
                f.write(f"Dictionary with {len(data)} entries:\n\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            
            elif isinstance(data, (list, tuple)):
                f.write(f"{type(data).__name__} with {len(data)} items:\n\n")
                for i, item in enumerate(data):
                    f.write(f"[{i}]: {item}\n")
            
            else:
                f.write(str(data))
        
        print(f"✓ Saved to text file: {output_path}")
        
    except Exception as e:
        print(f"✗ Error saving to {output_path}: {e}")

def save_to_json_file(data, output_path, description=""):
    """
    Save data to a JSON file if possible.
    
    Args:
        data: Data to save
        output_path: Path to save the JSON file
        description: Optional description
    """
    try:
        # Convert data to JSON-serializable format
        if isinstance(data, dict):
            json_data = {str(k): str(v) if not isinstance(v, (dict, list, int, float, bool, str, type(None))) else v 
                        for k, v in data.items()}
        else:
            json_data = data
        
        output_dict = {
            "description": description,
            "data_type": str(type(data)),
            "data": json_data
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to JSON file: {output_path}")
        
    except Exception as e:
        print(f"✗ Error saving JSON to {output_path}: {e}")

def depickle_and_save_metadata():
    """Load and save ONNX metadata."""
    metadata_path = 'models/onnx_metadata.pkl'
    metadata = depickle_file(metadata_path)
    
    if metadata:
        print("\n=== ONNX Metadata ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # Save to text file
        save_to_text_file(
            metadata, 
            'output/onnx_metadata.txt', 
            "ONNX Model Metadata"
        )
        
        # Save to JSON file
        save_to_json_file(
            metadata,
            'output/onnx_metadata.json',
            "ONNX Model Metadata"
        )
    
    return metadata

def depickle_and_save_mappings():
    """Load and save user and movie mappings."""
    user_map_path = 'models/user_map.pkl'
    movie_map_path = 'models/movie_map.pkl'
    
    print("\n=== User Mappings ===")
    user_map = depickle_file(user_map_path)
    if user_map:
        save_to_text_file(
            user_map,
            'output/user_mappings.txt',
            f"User ID Mappings ({len(user_map)} users)"
        )
        save_to_json_file(
            user_map,
            'output/user_mappings.json',
            f"User ID Mappings ({len(user_map)} users)"
        )
    
    print("\n=== Movie Mappings ===")
    movie_map = depickle_file(movie_map_path)
    if movie_map:
        save_to_text_file(
            movie_map,
            'output/movie_mappings.txt',
            f"Movie ID Mappings ({len(movie_map)} movies)"
        )
        save_to_json_file(
            movie_map,
            'output/movie_mappings.json',
            f"Movie ID Mappings ({len(movie_map)} movies)"
        )
    
    return user_map, movie_map

def explore_and_save_pickle_file(file_path):
    """
    Explore a pickle file and save its contents.
    
    Args:
        file_path: Path to the pickle file
    """
    data = depickle_file(file_path)
    
    if data is None:
        return
    
    print(f"\n=== Exploring {file_path} ===")
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if isinstance(data, dict):
        print("Available keys:")
        keys = list(data.keys())
        for i, key in enumerate(keys):
            print(f"  {i}: {key}")
        
        # Show some sample values
        for key in keys[:5]:
            value = data[key]
            if isinstance(value, (str, int, float, bool)):
                print(f"\n{key}: {value}")
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                print(f"\n{key}: {value}")
            else:
                print(f"\n{key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
    
    # Save to files
    save_to_text_file(
        data,
        f'output/{base_name}.txt',
        f"Contents of {file_path}"
    )
    
    save_to_json_file(
        data,
        f'output/{base_name}.json',
        f"Contents of {file_path}"
    )

def save_summary_report(pickle_files):
    """Create a summary report of all pickle files."""
    summary_path = 'output/pickle_summary.txt'
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("PICKLE FILES SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {os.path.getctime}\n")
            f.write(f"Total pickle files found: {len(pickle_files)}\n\n")
            
            for i, file in enumerate(pickle_files, 1):
                file_path = os.path.join('models', file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                f.write(f"{i}. {file}\n")
                f.write(f"   Path: {file_path}\n")
                f.write(f"   Size: {file_size:,} bytes ({file_size/1024:.2f} KB)\n")
                f.write(f"   Output files: output/{os.path.splitext(file)[0]}.txt, output/{os.path.splitext(file)[0]}.json\n")
                f.write("\n")
        
        print(f"✓ Summary report saved: {summary_path}")
        
    except Exception as e:
        print(f"✗ Error creating summary report: {e}")

if __name__ == "__main__":
    print("Depickling model files and saving to output directory...\n")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Check what pickle files exist in models directory
    models_dir = 'models'
    if os.path.exists(models_dir):
        pickle_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"Found pickle files in {models_dir}:")
        for i, file in enumerate(pickle_files):
            print(f"  {i}: {file}")
        
        print("\n" + "="*50)
        
        # Load and save specific files
        if 'onnx_metadata.pkl' in pickle_files:
            depickle_and_save_metadata()
        
        if 'user_map.pkl' in pickle_files and 'movie_map.pkl' in pickle_files:
            depickle_and_save_mappings()
        
        # Process all pickle files
        print("\n" + "="*50)
        print("Processing all pickle files...")
        
        for file in pickle_files:
            file_path = os.path.join(models_dir, file)
            explore_and_save_pickle_file(file_path)
            print("\n" + "-"*30)
        
        # Create summary report
        save_summary_report(pickle_files)
        
        print(f"\n🎉 All files processed!")
        print(f"Output files saved in: output/")
        print(f"Files created:")
        if os.path.exists('output'):
            output_files = os.listdir('output')
            for file in sorted(output_files):
                print(f"  - output/{file}")
    
    else:
        print(f"Models directory '{models_dir}' not found.")
        print("Available pickle files in current directory:")
        pickle_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        for file in pickle_files:
            explore_and_save_pickle_file(file)