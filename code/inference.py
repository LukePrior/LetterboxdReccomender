import pandas as pd
import numpy as np
import torch
import pickle
import os
from train import SASRecModel
from letterboxdpy.user import User, user_films_rated, user_films_watched

class MovieRecommender:
    def __init__(self, model_path='models/sasrec_model_best.pth', 
                 user_map_path='models/user_map.pkl', 
                 movie_map_path='models/movie_map.pkl',
                 ratings_path=None,
                 movies_metadata_path='data/movies.csv'):
        """
        Initialize the movie recommender.
        
        Args:
            model_path: Path to the trained model
            user_map_path: Path to user mapping pickle file
            movie_map_path: Path to movie mapping pickle file
            ratings_path: Path to the original ratings CSV file
            movies_metadata_path: Path to movies metadata CSV file for name matching
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load mappings
        self.user_map, self.movie_map = self.load_mappings(user_map_path, movie_map_path)
        
        # Create reverse mappings
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_movie_map = {v: k for k, v in self.movie_map.items()}
        
        # Load movie metadata for name matching and lookups
        self.movie_titles, self.movie_lookup = self.load_movie_titles(movies_metadata_path)
        
        # Load model
        self.model, self.model_params = self.load_model(model_path)
        
        print(f"Recommender initialized with {len(self.movie_map)} movies and {len(self.user_map)} users")
    
    def load_mappings(self, user_map_path, movie_map_path):
        """Load user and movie mappings from pickle files."""
        with open(user_map_path, 'rb') as f:
            user_map = pickle.load(f)
        
        with open(movie_map_path, 'rb') as f:
            movie_map = pickle.load(f)
        
        return user_map, movie_map
    
    def load_model(self, model_path):
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters
        model_params = checkpoint.get('model_params', {})
        
        # Initialize model with saved parameters
        model = SASRecModel(
            num_movies=model_params.get('num_movies', len(self.movie_map)),
            embed_dim=model_params.get('embed_dim', 128),
            num_heads=model_params.get('num_heads', 8),
            num_layers=model_params.get('num_layers', 2),
            max_len=model_params.get('max_len', 50)
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'val_accuracy' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['val_accuracy']:.4f}")
        
        return model, model_params
    
    def load_movie_titles(self, movies_metadata_path):
        """Load movie titles from metadata CSV for name matching and lookups."""
        movie_titles = {}
        movie_lookup = {}
        
        if movies_metadata_path and os.path.exists(movies_metadata_path):
            try:
                movies_df = pd.read_csv(movies_metadata_path)
                # Create mappings from movie_id to title
                for _, row in movies_df.iterrows():
                    movie_id = row['movieId']  # Assuming 'movieId' column exists
                    title = row.get('title', '')  # Assuming 'title' column exists
                    if movie_id in self.movie_map and title:
                        # For fuzzy matching (normalized)
                        normalized_title = self.normalize_title(title)
                        movie_titles[normalized_title] = movie_id
                        # For exact lookups (original title)
                        movie_lookup[movie_id] = title
                print(f"Loaded {len(movie_titles)} movie titles for matching")
                print(f"Loaded {len(movie_lookup)} movie titles for lookup")
                return movie_titles, movie_lookup
            except Exception as e:
                print(f"Warning: Could not load movie metadata: {e}")
                return {}, {}
        else:
            print("Warning: No movie metadata file provided for name matching")
            return {}, {}
    
    def get_movie_title(self, movie_id):
        """Get movie title by ID, return ID as string if title not found."""
        return self.movie_lookup.get(movie_id, f"Movie {movie_id}")
    
    def normalize_title(self, title):
        """Normalize movie title for fuzzy matching."""
        import re
        # Convert to lowercase, remove special characters, normalize spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def fetch_letterboxd_user_films(self, username, rating_limit=None):
        """
        Fetch watched/reviewed films for a Letterboxd user.
        
        Args:
            username: Letterboxd username
            limit: Maximum number of films to fetch (None for all)
        
        Returns:
            List of movie IDs that are in our dataset
        """
        try:
            # Get user's watched films
            user_instance = User(username)
            if rating_limit is None:
                print(f"Fetching watched films for Letterboxd user: {username}")
                user_films = user_films_watched(user_instance)
            else:
                print(f"Fetching reviewed films for Letterboxd user: {username}")
                user_films = user_films_rated(user_instance)
            print(f"Found {len(user_films)} films from Letterboxd")
            
            movie_ids = []
            processed_count = 0
            matched_count = 0
            
            for film in user_films:
                processed_count += 1
                if matched_count >= 50:
                    print("Reached limit of 50 matched films, stopping early")
                    break
                if rating_limit is not None and len(film[3]) < rating_limit:
                    continue
                
                # Extract film title - assuming film is a tuple (title, slug) based on your output
                film_title = film[0]
                
                # Normalize the title for matching
                normalized_title = self.normalize_title(film_title)
                
                # Try to find a match in our movie database
                matched_movie_id = None
                
                # Direct match
                if normalized_title in self.movie_titles:
                    matched_movie_id = self.movie_titles[normalized_title]
                else:
                    # Fuzzy matching - look for partial matches
                    for db_title, movie_id in self.movie_titles.items():
                        # Check if titles are similar (simple substring matching)
                        if (normalized_title in db_title or 
                            db_title in normalized_title or
                            self.titles_similar(normalized_title, db_title)):
                            matched_movie_id = movie_id
                            break
                
                if matched_movie_id:
                    movie_ids.append(matched_movie_id)
                    matched_count += 1
                
                # Progress indicator for large collections
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} films, matched {matched_count} in dataset")
            
            print(f"Fetched {processed_count} films, {matched_count} matched in dataset")
            filtered_movie_ids = list(reversed(movie_ids[:50]))
            return filtered_movie_ids
            
        except Exception as e:
            print(f"Error fetching Letterboxd data for user {username}: {str(e)}")
            return []
    
    def titles_similar(self, title1, title2, threshold=0.8):
        """
        Check if two titles are similar using simple similarity measure.
        
        Args:
            title1, title2: Normalized titles to compare
            threshold: Similarity threshold (0-1)
        
        Returns:
            bool: True if titles are similar enough
        """
        # Simple Jaccard similarity using words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
    
    def get_letterboxd_recommendations(self, username, top_k=10, rating_limit=None):
        """
        Get movie recommendations for a Letterboxd user.
        
        Args:
            username: Letterboxd username
            top_k: Number of recommendations to return
            rating_limit: Minimum rating limit for films (None for no limit)
        
        Returns:
            List of tuples: (movie_title, original_movie_id, score)
        """
        # Fetch user's watched films
        user_films = self.fetch_letterboxd_user_films(username, rating_limit=rating_limit)
        
        if not user_films:
            print(f"No films found for user {username} that match our dataset")
            return []
        
        print(f"Using {len(user_films)} films from {username}'s Letterboxd for recommendations")
        
        # Get recommendations based on user's film history
        return self.get_recommendations(user_films, top_k=top_k, exclude_seen=True)
    
    def prepare_sequence(self, movie_history, max_len=None):
        """
        Prepare a movie sequence for model input.
        
        Args:
            movie_history: List of movie IDs (already mapped)
            max_len: Maximum sequence length (uses model default if None)
        
        Returns:
            torch.Tensor: Padded sequence ready for model input
        """
        if max_len is None:
            max_len = self.model_params.get('max_len', 50)
        
        # Take the last max_len movies or pad with zeros
        if len(movie_history) > max_len:
            sequence = movie_history[-max_len:]
        else:
            # Left-padding with zeros
            sequence = [0] * (max_len - len(movie_history)) + movie_history
        
        return torch.tensor([sequence], dtype=torch.long).to(self.device)
    
    def get_recommendations(self, movie_history, top_k=10, exclude_seen=True):
        """
        Get movie recommendations based on viewing history.
        
        Args:
            movie_history: List of movie IDs (original IDs, not mapped)
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude movies already in history
        
        Returns:
            List of tuples: (movie_title, original_movie_id, score)
        """
        # Map movie IDs to model vocabulary
        mapped_history = []
        for movie_id in movie_history:
            if movie_id in self.movie_map:
                mapped_history.append(self.movie_map[movie_id])
        
        if not mapped_history:
            print("Warning: No valid movies found in history")
            return []
        
        # Prepare input sequence
        input_sequence = self.prepare_sequence(mapped_history)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_sequence)
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        scores, predicted_indices = torch.topk(probabilities[0], k=min(top_k * 2, len(self.movie_map)))
        
        recommendations = []
        seen_movies = set(movie_history) if exclude_seen else set()
        
        for idx, score in zip(predicted_indices, scores):
            mapped_movie_id = idx.item()
            
            # Skip padding token (0)
            if mapped_movie_id == 0:
                continue
            
            # Convert back to original movie ID
            if mapped_movie_id in self.reverse_movie_map:
                original_movie_id = self.reverse_movie_map[mapped_movie_id]
                
                # Skip if already seen
                if exclude_seen and original_movie_id in seen_movies:
                    continue
                
                # Get movie title
                movie_title = self.get_movie_title(original_movie_id)
                recommendations.append((movie_title, original_movie_id, score.item()))
                
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    def recommend_similar_to_movie(self, movie_id, top_k=10):
        """
        Get movies similar to a given movie by using it as a single-item history.
        
        Args:
            movie_id: Original movie ID
            top_k: Number of recommendations to return
        
        Returns:
            List of tuples: (movie_title, original_movie_id, score)
        """
        return self.get_recommendations([movie_id], top_k, exclude_seen=True)
    
    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_config': self.model_params,
            'vocabulary_size': len(self.movie_map),
            'num_users': len(self.user_map),
            'device': str(self.device)
        }

def main():
    """Example usage of the MovieRecommender."""
    
    # Initialize recommender
    # Adjust paths as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_path = os.path.join(os.path.dirname(script_dir), 'data', 'qualifying_users_ratings.csv')
    movies_metadata_path = os.path.join(os.path.dirname(script_dir), 'data', 'movies.csv')
    
    recommender = MovieRecommender(
        model_path='models/sasrec_model_epoch_6.pth',
        user_map_path='models/user_map.pkl',
        movie_map_path='models/movie_map.pkl',
        movies_metadata_path=movies_metadata_path
    )
    
    print("\n" + "="*50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("="*50)
    
    # Display model info
    model_info = recommender.get_model_info()
    print(f"\nModel Info:")
    print(f"  Parameters: {model_info['model_parameters']:,}")
    print(f"  Vocabulary Size: {model_info['vocabulary_size']:,}")
    print(f"  Users: {model_info['num_users']:,}")
    print(f"  Device: {model_info['device']}")
    
    while True:
        # Example 1: Letterboxd user recommendations
        print(f"\n" + "-"*30)
        print("LETTERBOXD USER RECOMMENDATIONS")
        print("-"*30)
        
        letterboxd_username = input("Enter Letterboxd username (or press Enter to skip): ").strip()
        rating_limit = input("Enter minimum rating limit for films (or press Enter for no limit): ").strip()
        rating_limit = float(rating_limit) if len(rating_limit) > 0 else None
        
        if letterboxd_username:
            print(f"\nGetting recommendations for Letterboxd user: {letterboxd_username}")
            letterboxd_recs = recommender.get_letterboxd_recommendations(
                letterboxd_username, 
                top_k=10,
                rating_limit=rating_limit
            )
            
            if letterboxd_recs:
                print("Letterboxd Recommendations:")
                for i, (title, movie_id, score) in enumerate(letterboxd_recs, 1):
                    print(f"  {i}. {title} (ID: {movie_id}, score: {score:.4f})")
            else:
                print("No recommendations available for this user")
    
        else:
            break


if __name__ == "__main__":
    main()