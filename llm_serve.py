
# llm_serve.py

import os
import logging
from typing import List
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
from time import perf_counter

import fire
import torch
import uvicorn
from fastapi import FastAPI, APIRouter
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompletionRequest(BaseModel):
    prompt: str
    echo: bool = False
    max_length: int = 256
    temperature: float = 0.7

class CompletionResponse(BaseModel):
    text: str
    object: str = "text-completion"
    num_tokens: int # number of tokens generated
    elapsed: float # total elapsed time
    tps: float # tokens per second

class EmbeddingRequest(BaseModel):
    input: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[List[float]]
    object: str = "embedding"

def get_model(model_string: str, dtype: str, device_string: str) -> AutoModelForCausalLM:
    '''
    model_string: string of model to load
    dtype: storch dtype to use. One of [int8, bf16, fp16]
    '''
    accepted_dtype = ["int8", "bf16", "fp16"]
    assert dtype in accepted_dtype, f"Expected dtype to be one of {accepted_dtype}"
    logger.info("creating model=%s of dtype=%s on device=%s", model_string, dtype, device_string)
    if dtype == 'int8':
        model = AutoModelForCausalLM.from_pretrained(model_string, load_in_8bit=True)
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(model_string, torch_dtype=torch.bfloat16)
    elif dtype == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(model_string, torch_dtype=torch.float16)

    device = torch.device(device_string)
    if dtype in ['bf16', 'fp16']:
        model = model.to(device)
    logger.info("model creation suceeded.")
    model.eval()

    return model

def get_tokenizer(model_string: str) -> AutoTokenizer:
    logger.info("creating tokenizer.")
    return AutoTokenizer.from_pretrained(model_string, use_fast=False)

def prepare_llama_prompt(prompt:str) -> str:
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def prepare_model_input(prompt:str) -> list[str]:
    return tokenizer.encode(prompt, add_special_tokens=True)

def generate(prompt: str, max_length: int = 256, temperature: float = 0.7, echo = False) -> dict:

    prompt = prepare_llama_prompt(prompt)
    input_ids = prepare_model_input(prompt)
    init_len = len(input_ids)
    
    elapsed = []
    for ix in range(max_length):
        with torch.no_grad():
            tick = perf_counter()
            out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=False)
            probs = torch.softmax(out.logits[:, -1, :].squeeze()/temperature, axis=0)
            out_token_id = torch.multinomial(probs, 1)[0].item()
            elapsed.append(perf_counter() - tick)
        if out_token_id == tokenizer.eos_token_id:
            break;
        else:
            input_ids.append(out_token_id)
    
    if echo:
        num_tokens = len(input_ids) - init_len
    else:
        input_ids = input_ids[init_len:]
        num_tokens = len(input_ids)
    
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    tps = len(elapsed) / sum(elapsed) if len(elapsed) > 0 else 0
    return dict(text=text,
                num_tokens = num_tokens,
                tps = tps,
                elapsed = sum(elapsed))

def embed(inputs: List[str]) -> List[List[float]]:
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(inputs, return_tensors='pt', padding=True)
    encoded = encoded.to(model.device)
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True, use_cache=False)
    
    embeddings_np = output['hidden_states'][-1][:, -1, :].to(torch.float32).cpu().numpy()
    # convert to list of lists
    embeddings = []
    for emb in embeddings_np:
        embeddings.append(emb.tolist())
    
    torch.cuda.empty_cache()
    del(encoded)
    del(output)

    return dict(embedding=embeddings)

def create_logger(logdir=None):
    """
    creates & returns a logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    
    if logdir is not None:
        logdir = Path(logdir)
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)

        file_name = 'llm_serve_' + datetime.now().strftime("%y%m%d-%H%M")+ '.log'
        f_handler = logging.FileHandler(logdir / file_name)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


router = APIRouter()

@router.get("/")
async def check_server():
    return {"status": "ok", 
            "model": model.name_or_path, 
            "device": str(model.device), 
            "model_dtype":str(model.dtype)}

@router.post("/completion", response_model=CompletionResponse)
async def get_completion(input: CompletionRequest) -> CompletionResponse:
    try:
        response_dict =  generate(prompt=input.prompt, 
                                max_length=input.max_length, 
                                temperature=input.temperature, 
                                echo=input.echo)
        return CompletionResponse(**response_dict)
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise e

@router.post("/embedding", response_model=EmbeddingResponse)
async def get_embedding(input: EmbeddingRequest) -> EmbeddingResponse:
    try:
        response_dict =  embed(inputs=input.input)
        return EmbeddingResponse(**response_dict)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise e

def main(model_string: str = "meta-llama/Llama-2-7b-chat-hf",
         device_string: str = "cuda",
         dtype: str = 'int8',
         logdir: str = '/tmp',
         ip: str = "0.0.0.0",
         port: int = 8000):

    global logger, tokenizer, model
    
    app = FastAPI()
    app.include_router(router)

    logger = create_logger(logdir)
    tokenizer = get_tokenizer(model_string)
    model = get_model(model_string, dtype=dtype, device_string=device_string)

    uvicorn.run(app, host=ip, port=port)

if __name__ == "__main__":
    fire.Fire(main)


'''
python3 fastAPI/llm_serve.py \
    --model_string=mistralai/Mistral-7B-Instruct-v0.1 \
    --dtype=int8 \
    --device=cuda \
    --ip=0.0.0.0 \
    --port=8000
'''