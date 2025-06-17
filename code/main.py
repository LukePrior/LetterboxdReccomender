import pandas as pd
import numpy as np
import os

def process_ratings(csv_file=None):
    """
    Process ratings CSV to find users with >20 entries above their 70th percentile,
    returned in chronological order. Only includes movies with at least 10 reviews.
    """
    
    if csv_file is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level and into data directory
        csv_file = os.path.join(os.path.dirname(script_dir), 'data', 'ratings.csv')
    
    # Load the CSV file
    print(f"Loading CSV file from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime for filtering and sorting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Filter to only include reviews from 2010 onwards
    df = df[df['datetime'].dt.year >= 2010]
    
    print(f"Loaded {len(df)} ratings from 2010 onwards for {df['userId'].nunique()} users")
    
    # Filter to only include movies with at least 10 reviews
    movie_review_counts = df.groupby('movieId').size()
    movies_with_min_reviews = movie_review_counts[movie_review_counts >= 10].index
    df = df[df['movieId'].isin(movies_with_min_reviews)]
    
    print(f"After filtering to movies with ≥10 reviews: {len(df)} ratings for {df['movieId'].nunique()} movies and {df['userId'].nunique()} users")
    
    # Calculate 70th percentile for each user
    print("Calculating 70th percentiles...")
    user_percentiles = df.groupby('userId')['rating'].quantile(0.7).reset_index()
    user_percentiles.columns = ['userId', 'percentile_70']
    
    # Merge back with original data
    df_with_percentiles = df.merge(user_percentiles, on='userId')
    
    # Find ratings above each user's 70th percentile
    df_above_percentile = df_with_percentiles[
        df_with_percentiles['rating'] >= df_with_percentiles['percentile_70']
    ]
    
    # Count entries above percentile for each user
    user_counts = df_above_percentile.groupby('userId').size().reset_index(name='count_above_percentile')
    
    # Filter users with more than 20 entries above their 70th percentile
    qualifying_users = user_counts[user_counts['count_above_percentile'] > 20]['userId'].tolist()
    
    print(f"Found {len(qualifying_users)} users with >20 ratings above their 70th percentile")
    
    # Get only ratings at or above 70th percentile for qualifying users and sort chronologically
    result = df_above_percentile[df_above_percentile['userId'].isin(qualifying_users)].copy()
    result = result.sort_values(['userId', 'timestamp'])
    
    # Remove the percentile_70 and datetime columns from result
    result = result.drop(['percentile_70', 'datetime'], axis=1)
    
    return result, qualifying_users

def save_results(result_df, qualifying_users, output_file='qualifying_users_ratings.csv'):
    """Save results to CSV file"""
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Qualifying users: {len(qualifying_users)}")
    print(f"Total ratings at or above 70th percentile for these users: {len(result_df)}")
    
    # Show sample of results
    print(f"\nFirst 10 rows:")
    print(result_df[['userId', 'movieId', 'rating', 'timestamp']].head(10))

def calculate_movie_review_percentiles(result_df):
    """
    Calculate the 10, 20, 30, etc percentiles for number of reviews per movie
    from the filtered results.
    """
    print("\nCalculating movie review count percentiles...")
    
    # Count number of reviews per movie
    movie_review_counts = result_df.groupby('movieId').size().reset_index(name='review_count')
    
    print(f"Found {len(movie_review_counts)} unique movies in the filtered dataset")
    print(f"Total reviews across all movies: {movie_review_counts['review_count'].sum()}")
    
    # Calculate percentiles from 10 to 100 in increments of 10
    percentiles = range(10, 101, 10)
    percentile_values = []
    
    for p in percentiles:
        value = np.percentile(movie_review_counts['review_count'], p)
        percentile_values.append(value)
        print(f"{p}th percentile: {value:.1f} reviews")
    
    # Create a summary dataframe
    percentile_summary = pd.DataFrame({
        'Percentile': percentiles,
        'Review_Count': percentile_values
    })
    
    # Additional statistics
    print(f"\nAdditional statistics:")
    print(f"Mean reviews per movie: {movie_review_counts['review_count'].mean():.2f}")
    print(f"Median reviews per movie: {movie_review_counts['review_count'].median():.1f}")
    print(f"Minimum reviews per movie: {movie_review_counts['review_count'].min()}")
    print(f"Maximum reviews per movie: {movie_review_counts['review_count'].max()}")
    print(f"Standard deviation: {movie_review_counts['review_count'].std():.2f}")
    
    return percentile_summary, movie_review_counts

if __name__ == "__main__":
    # Process the ratings
    results, users = process_ratings()
    
    # Calculate movie review percentiles
    percentile_summary, movie_counts = calculate_movie_review_percentiles(results)
    
    # Save results
    save_results(results, users)