# VQVAE+ 

WIP...

### LLM Serve
    python3 llm_serve.py \
        --model_string=mistralai/Mistral-7B-Instruct-v0.1 \
        --dtype=int8 \
        --device=cuda \
        --ip=0.0.0.0 \
        --port=8000
Python functions to interact w/ endpoints are available at:
    
    from data.llm_requests import get_completion, get_embedding




### Generate Image Captions Embeddings for Celeb-A


1. The opt-1.3b model is used for embedding, which creates 2048 dimensional embeddings.  
    Start server for getting embeddings.
    
        python3 llm_serve.py --model_string=facebook/opt-1.3b --dtype=int8


2. Create and save Embeddings for Celeb-A using provided methods.

        from data.llm_requests import get_embedding
        
        dataset_path = Path("<...Path to Celeb-A dataset...>")
        df = get_caption_embedding(dataset_path)
        
        df.to_csv(dataset_path / 'caption_embedding.csv', index=Fals
