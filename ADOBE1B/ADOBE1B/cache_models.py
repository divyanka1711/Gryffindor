

import os
from sentence_transformers import SentenceTransformer

def cache_models():
   
    print("Start model caching process...")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download and cache the sentence transformer model
        print("Download SentenceTransformer model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test the model to ensure it's working
        test_sentence = "This is a test sentence."
        embedding = model.encode(test_sentence)
        print(f"Model loaded successfull. Test embedding shape: {embedding.shape}")
        
        print("Model caching completed successfull!")
        
    except Exception as e:
        print(f"Error during model caching: {e}")
        raise

if __name__ == "__main__":
    cache_models()
