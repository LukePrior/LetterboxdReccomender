import json
import csv
import os

def create_movie_id_to_name_mapping():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to the project root (one level up from code directory)
    project_root = os.path.dirname(script_dir)
    movie_map_path = os.path.join(project_root, 'output', 'movie_map.json')
    movies_csv_path = os.path.join(project_root, 'data', 'movies.csv')
    output_path = os.path.join(project_root, 'output', 'movie_id_to_name.json')
    
    print(f"Reading movie mapping from: {movie_map_path}")
    print(f"Reading movies data from: {movies_csv_path}")
    
    # Read the movie mapping from movie_map.json
    with open(movie_map_path, 'r') as f:
        movie_map = json.load(f)
    
    # Create reverse mapping: external_id -> internal_id
    external_to_internal = movie_map['data']
    
    # Read movies.csv and create internal_id -> name mapping
    internal_id_to_name = {}
    
    with open(movies_csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            external_id = row['movieId']
            title = row['title']
            
            # Check if this external ID exists in our mapping
            if external_id in external_to_internal:
                internal_id = external_to_internal[external_id]
                internal_id_to_name[internal_id] = title
    
    # Create the final JSON structure
    result = {
        "data": internal_id_to_name
    }
    
    # Write to output file
    print(f"Writing output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Created movie_id_to_name.json with {len(internal_id_to_name)} movies")
    return result

if __name__ == "__main__":
    create_movie_id_to_name_mapping()