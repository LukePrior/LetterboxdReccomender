import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SASRecModel(nn.Module):
    def __init__(self, num_movies, embed_dim=128, num_heads=8, num_layers=2, max_len=50):
        super(SASRecModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Item embedding layer with padding_idx=0
        self.item_embedding = nn.Embedding(
            num_embeddings=num_movies + 1,  # +1 for padding token
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, num_movies + 1)
    
    def forward(self, src):
        # src shape: (batch_size, seq_len)
        batch_size, seq_len = src.shape
        
        # Create padding mask (True for padding tokens)
        src_key_padding_mask = (src == 0)
        
        # Item embeddings
        embeddings = self.item_embedding(src)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, embed_dim)
        
        # Get the last item's representation
        # We want the last non-padding token for each sequence
        last_item_output = transformer_output[:, -1, :]  # (batch_size, embed_dim)
        
        # Project to vocabulary size
        logits = self.output_layer(last_item_output)  # (batch_size, num_movies + 1)
        
        return logits

class MovieRecommendationDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

def load_and_preprocess_data(csv_file=None, max_len=50, min_len=10):
    """
    Load and preprocess the ratings data according to the specified format.
    
    Args:
        csv_file: Path to the CSV file. If None, uses the default path.
        max_len: Maximum sequence length for padding.
        min_len: Minimum sequence length for creating training samples.
    
    Returns:
        sequences: List of padded input sequences
        targets: List of target items
        user_map: Dictionary mapping original userIds to continuous integers
        movie_map: Dictionary mapping original movieIds to continuous integers (starting from 1)
        num_users: Number of unique users
        num_movies: Number of unique movies
    """
    
    if csv_file is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level and into data directory
        csv_file = os.path.join(os.path.dirname(script_dir), 'data', 'qualifying_users_ratings.csv')
    
    print(f"Loading data from: {csv_file}")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime for filtering
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort chronologically by userId and then timestamp
    df = df.sort_values(['userId', 'timestamp'])
    
    # Create entity mappings
    unique_users = df['userId'].unique()
    unique_movies = df['movieId'].unique()
    
    # User mapping: 0 to N-1
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # Movie mapping: 1 to M (0 is reserved for padding)
    movie_map = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
    
    num_users = len(unique_users)
    num_movies = len(unique_movies)
    
    print(f"Created mappings: {num_users} users, {num_movies} movies")
    print(f"Movie IDs mapped to range [1, {num_movies}], 0 reserved for padding")
    
    # Map the original IDs to new continuous IDs
    df['user_id_mapped'] = df['userId'].map(user_map)
    df['movie_id_mapped'] = df['movieId'].map(movie_map)
    
    # Generate user histories with minimum length filter
    print(f"Generating user histories (min length: {min_len + 1})...")
    
    # Group by user and filter by minimum length in one operation
    user_groups = df.groupby('user_id_mapped')['movie_id_mapped']
    user_histories = [
        group.tolist() 
        for _, group in user_groups 
        if len(group) > min_len
    ]
    
    print(f"Generated histories for {len(user_histories)} users with >{min_len} interactions")
    
    # Create training samples using sliding window
    print("Creating training samples with sliding window...")
    sequences = []
    targets = []
    
    for history in user_histories:
        # Start from min_len to ensure we have at least min_len items in input sequence
        for i in range(min_len, len(history)):
            # Input sequence: all items up to position i
            input_seq = history[:i]
            # Target: item at position i
            target = history[i]
            
            # Pad the input sequence
            if len(input_seq) > max_len:
                # If sequence is longer than max_len, take the last max_len items
                input_seq = input_seq[-max_len:]
            else:
                # Left-padding with 0s
                input_seq = [0] * (max_len - len(input_seq)) + input_seq
            
            sequences.append(input_seq)
            targets.append(target)
    
    print(f"Created {len(sequences)} training samples")
    print(f"Sequence length: {max_len} (padded), minimum non-padded length: {min_len}")
    print(f"Target range: [1, {num_movies}]")
    
    return sequences, targets, user_map, movie_map, num_users, num_movies

def save_mappings(user_map, movie_map, output_dir='models'):
    """Save the user and movie mappings for later use during inference."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'user_map.pkl'), 'wb') as f:
        pickle.dump(user_map, f)
    
    with open(os.path.join(output_dir, 'movie_map.pkl'), 'wb') as f:
        pickle.dump(movie_map, f)
    
    print(f"Mappings saved to {output_dir}/")

def create_data_loader(sequences, targets, batch_size=512, shuffle=True):
    """Create a PyTorch DataLoader from the sequences and targets."""
    dataset = MovieRecommendationDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_train_val_split(sequences, targets, val_split=0.2, random_state=42):
    """
    Split the data into training and validation sets.
    
    Args:
        sequences: List of input sequences
        targets: List of target items
        val_split: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        train_sequences, val_sequences, train_targets, val_targets
    """
    train_seq, val_seq, train_tgt, val_tgt = train_test_split(
        sequences, targets, 
        test_size=val_split, 
        random_state=random_state,
        stratify=None  # Can't stratify with this many classes
    )
    
    print(f"Training samples: {len(train_seq)}")
    print(f"Validation samples: {len(val_seq)}")
    
    return train_seq, val_seq, train_tgt, val_tgt

def evaluate_model(model, val_loader, num_movies, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The trained model
        val_loader: Validation DataLoader
        num_movies: Number of unique movies
        device: Device to run evaluation on
    
    Returns:
        avg_val_loss: Average validation loss
        accuracy: Top-1 accuracy
        top5_accuracy: Top-5 accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0.0
    correct_predictions = 0
    top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(sequences)
            
            # Calculate loss
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Calculate accuracy metrics
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == targets).sum().item()
            
            # Calculate top-5 accuracy
            _, top5_pred = torch.topk(logits, 5, dim=1)
            top5_correct += sum([targets[i] in top5_pred[i] for i in range(len(targets))])
            
            total_samples += targets.size(0)
    
    avg_val_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    top5_accuracy = top5_correct / total_samples
    
    return avg_val_loss, accuracy, top5_accuracy

def train_model(train_loader, num_movies, num_epochs=10, embed_dim=128, num_heads=8, num_layers=2, max_len=50, lr=0.001, device=None):
    """
    Train the SASRec model.
    
    Args:
        train_loader: DataLoader with training data
        num_movies: Number of unique movies (vocabulary size)
        num_epochs: Number of training epochs
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_len: Maximum sequence length
        lr: Learning rate
        device: Device to train on (cuda/cpu)
    
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Initialize model
    model = SASRecModel(
        num_movies=num_movies,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens in loss
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")
    
    return model, train_losses

def train_model_with_validation(train_loader, val_loader, num_movies, num_epochs=5, embed_dim=128, 
                               num_heads=8, num_layers=2, max_len=50, lr=0.001, device=None,
                               patience=3, min_delta=0.001, save_every_epoch=True, output_dir='models'):
    """
    Train the SASRec model with validation monitoring and early stopping.
    
    Args:
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        num_movies: Number of unique movies (vocabulary size)
        num_epochs: Maximum number of training epochs
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_len: Maximum sequence length
        lr: Learning rate
        device: Device to train on (cuda/cpu)
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in validation loss to qualify as improvement
        save_every_epoch: Whether to save model at each epoch
        output_dir: Directory to save models
    
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = SASRecModel(
        num_movies=num_movies,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_top5_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Training...")
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}], Loss: {loss.item():.4f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validating...")
        val_loss, val_acc, val_top5_acc = evaluate_model(model, val_loader, num_movies, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_top5_accuracies.append(val_top5_acc)
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Val Top-5 Accuracy: {val_top5_acc:.4f} ({val_top5_acc*100:.2f}%)")
        
        # Save model at each epoch if requested
        if save_every_epoch:
            epoch_model_path = os.path.join(output_dir, f'sasrec_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_top5_accuracy': val_top5_acc,
                'model_params': {
                    'num_movies': num_movies,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'max_len': max_len
                }
            }, epoch_model_path)
            print(f"  Model saved: {epoch_model_path}")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            print(f"  ✓ New best validation loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    # If we completed all epochs without early stopping, use the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nTraining completed. Using best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        
        # Save the best model separately
        best_model_path = os.path.join(output_dir, 'sasrec_model_best.pth')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_loss': best_val_loss,
            'model_params': {
                'num_movies': num_movies,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'max_len': max_len
            }
        }, best_model_path)
        print(f"Best model saved: {best_model_path}")
    
    return model, train_losses, val_losses, val_accuracies, val_top5_accuracies

def save_model(model, output_dir='models', model_name='sasrec_model.pth', epoch=None, 
               train_loss=None, val_loss=None, val_accuracy=None, model_params=None):
    """Save the trained model with additional metadata."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    # Add optional metadata
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    if val_accuracy is not None:
        checkpoint['val_accuracy'] = val_accuracy
    if model_params is not None:
        checkpoint['model_params'] = model_params
    
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Configuration
    MAX_LEN = 50
    MIN_LEN = 10
    BATCH_SIZE = 1024
    VAL_SPLIT = 0.2  # 20% for validation
    
    # Model hyperparameters
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Early stopping parameters
    PATIENCE = 3
    MIN_DELTA = 0.001
    
    print("Starting data preprocessing...")
    
    # Load and preprocess data
    sequences, targets, user_map, movie_map, num_users, num_movies = load_and_preprocess_data(
        max_len=MAX_LEN, min_len=MIN_LEN
    )
    
    # Create train/validation split
    print(f"\nCreating train/validation split ({VAL_SPLIT*100:.0f}% validation)...")
    train_seq, val_seq, train_tgt, val_tgt = create_train_val_split(
        sequences, targets, val_split=VAL_SPLIT
    )
    
    # Save mappings
    save_mappings(user_map, movie_map)
    
    # Create data loaders
    train_loader = create_data_loader(train_seq, train_tgt, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_seq, val_tgt, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate new batch counts
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    print(f"\nData loaders created:")
    print(f"  Training batches: {train_batches}")
    print(f"  Validation batches: {val_batches}")
    
    print(f"\nStarting model training with validation...")
    print(f"Configuration:")
    print(f"  - Embedding dimension: {EMBED_DIM}")
    print(f"  - Number of attention heads: {NUM_HEADS}")
    print(f"  - Number of transformer layers: {NUM_LAYERS}")
    print(f"  - Max epochs: {NUM_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Vocabulary size: {num_movies + 1}")
    print(f"  - Early stopping patience: {PATIENCE}")
    
    # Train the model with validation
    model, train_losses, val_losses, val_accuracies, val_top5_accuracies = train_model_with_validation(
        train_loader=train_loader,
        val_loader=val_loader,
        num_movies=num_movies,
        num_epochs=NUM_EPOCHS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA
    )
    
    # Save the trained model
    save_model(model)
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_top5_accuracies': val_top5_accuracies
    }
    
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    
    print("\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f} ({val_accuracies[-1]*100:.2f}%)")
    print(f"Final validation top-5 accuracy: {val_top5_accuracies[-1]:.4f} ({val_top5_accuracies[-1]*100:.2f}%)")
    print("Training history saved to models/training_history.pkl")