import torch
import torch.onnx
import pickle
import os
import numpy as np
from train import SASRecModel

def load_model_and_mappings(model_path='models/sasrec_model_epoch_6.pth', 
                           user_map_path='models/user_map.pkl', 
                           movie_map_path='models/movie_map.pkl'):
    """
    Load the trained model and mappings.
    
    Args:
        model_path: Path to the saved PyTorch model
        user_map_path: Path to user mappings
        movie_map_path: Path to movie mappings
    
    Returns:
        model: Loaded PyTorch model
        user_map: User ID mappings
        movie_map: Movie ID mappings
        model_params: Model parameters
    """
    print(f"Loading model from: {model_path}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model_params = checkpoint['model_params']
    
    # Initialize the model
    model = SASRecModel(
        num_movies=model_params['num_movies'],
        embed_dim=model_params['embed_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        max_len=model_params['max_len']
    )
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load mappings
    with open(user_map_path, 'rb') as f:
        user_map = pickle.load(f)
    
    with open(movie_map_path, 'rb') as f:
        movie_map = pickle.load(f)
    
    print(f"Mappings loaded: {len(user_map)} users, {len(movie_map)} movies")
    
    return model, user_map, movie_map, model_params

def convert_to_onnx(model, model_params, output_path='models/sasrec_model.onnx', dynamic_batch=True):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        model_params: Model parameters dictionary
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_batch: Whether to support dynamic batch sizes
    
    Returns:
        bool: True if conversion successful
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input with the expected shape
        max_len = model_params['max_len']
        batch_size = 1
        
        # Create example input (batch_size, seq_len)
        dummy_input = torch.randint(1, model_params['num_movies'] + 1, (batch_size, max_len), dtype=torch.long)
        
        print(f"Converting model to ONNX...")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input dtype: {dummy_input.dtype}")
        
        # Define input and output names
        input_names = ['input_sequence']
        output_names = ['output_logits']
        
        # Define dynamic axes if dynamic batch is enabled
        if dynamic_batch:
            dynamic_axes = {
                'input_sequence': {0: 'batch_size'},
                'output_logits': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False        
        )
        
        print(f"✓ Model successfully converted to ONNX: {output_path}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
            
            # Print model info
            print(f"ONNX model info:")
            print(f"  - Inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"  - Outputs: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            print("⚠ ONNX package not found. Install with: pip install onnx")
        except Exception as e:
            print(f"⚠ ONNX model verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error converting model to ONNX: {e}")
        return False

def test_onnx_model(pytorch_model, onnx_path, model_params, num_test_samples=5):
    """
    Test the ONNX model against PyTorch model to ensure correctness.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        model_params: Model parameters
        num_test_samples: Number of test samples to compare
    
    Returns:
        bool: True if all tests pass
    """
    try:
        import onnxruntime as ort
        
        print(f"\nTesting ONNX model against PyTorch model...")
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        max_len = model_params['max_len']
        num_movies = model_params['num_movies']
        
        all_tests_passed = True
        
        for i in range(num_test_samples):
            # Create random test input
            test_input = torch.randint(1, num_movies + 1, (1, max_len), dtype=torch.long)
            
            # PyTorch inference
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)
            
            # ONNX inference
            onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]
            
            # Compare outputs
            pytorch_numpy = pytorch_output.numpy()
            max_diff = np.max(np.abs(pytorch_numpy - onnx_output))
            
            if max_diff < 1e-5:
                print(f"  Test {i+1}/{num_test_samples}: ✓ PASSED (max_diff: {max_diff:.2e})")
            else:
                print(f"  Test {i+1}/{num_test_samples}: ✗ FAILED (max_diff: {max_diff:.2e})")
                all_tests_passed = False
        
        if all_tests_passed:
            print("✓ All ONNX tests passed!")
        else:
            print("✗ Some ONNX tests failed!")
        
        return all_tests_passed
        
    except ImportError:
        print("⚠ ONNX Runtime not found. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"✗ Error testing ONNX model: {e}")
        return False

def save_conversion_metadata(output_dir, model_params, user_map, movie_map):
    """
    Save metadata needed for ONNX model inference.
    
    Args:
        output_dir: Directory to save metadata
        model_params: Model parameters
        user_map: User mappings
        movie_map: Movie mappings
    """
    metadata = {
        'model_params': model_params,
        'num_users': len(user_map),
        'num_movies': len(movie_map),
        'input_shape': [1, model_params['max_len']],
        'input_dtype': 'int64',
        'output_shape': [1, model_params['num_movies'] + 1],
        'output_dtype': 'float32',
        'description': 'SASRec model for movie recommendation'
    }
    
    metadata_path = os.path.join(output_dir, 'onnx_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Conversion metadata saved: {metadata_path}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'models/sasrec_model_epoch_6.pth'  # or 'models/sasrec_model.pth'
    ONNX_OUTPUT_PATH = 'models/sasrec_model.onnx'
    USER_MAP_PATH = 'models/user_map.pkl'
    MOVIE_MAP_PATH = 'models/movie_map.pkl'
    
    # ONNX export settings

    RUN_TESTS = True  # Test ONNX model against PyTorch
    
    print("Starting PyTorch to ONNX conversion...")
    print(f"Configuration:")
    print(f"  - Model path: {MODEL_PATH}")
    print(f"  - ONNX output: {ONNX_OUTPUT_PATH}")

    
    # Check if required files exist
    required_files = [MODEL_PATH, USER_MAP_PATH, MOVIE_MAP_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"✗ Required file not found: {file_path}")
            print("Please ensure you have trained the model first using train.py")
            exit(1)
    
    try:
        # Load model and mappings
        model, user_map, movie_map, model_params = load_model_and_mappings(
            MODEL_PATH, USER_MAP_PATH, MOVIE_MAP_PATH
        )
        
        # Create output directory
        output_dir = os.path.dirname(ONNX_OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to ONNX
        success = convert_to_onnx(
            model, model_params, ONNX_OUTPUT_PATH
        )
        
        if success:
            # Save conversion metadata
            save_conversion_metadata(output_dir, model_params, user_map, movie_map)
            
            # Test ONNX model if requested
            if RUN_TESTS:
                test_onnx_model(model, ONNX_OUTPUT_PATH, model_params)
            
            print(f"\n🎉 Conversion completed successfully!")
            print(f"ONNX model saved to: {ONNX_OUTPUT_PATH}")
            print(f"Model size: {os.path.getsize(ONNX_OUTPUT_PATH) / (1024*1024):.2f} MB")
            
        else:
            print("✗ Conversion failed!")
            exit(1)
            
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        exit(1)