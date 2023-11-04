
from typing import List
import requests

BASE_URI = 'http://localhost:8000'

headers = {
    "Content-type": "application/json",
    "accept": "application/json"
}

def get_completion(prompt:str, 
                   max_length:int = 256,
                   echo:bool = False,
                   temperature: float = 0.7
                   ) -> dict:
    completion_payload = dict(
        prompt=prompt,
        echo=echo,
        max_length=max_length,
        temperature=temperature)
    response = requests.post(
        f"{BASE_URI}/completion",
        headers=headers,
        json=completion_payload)
    return response.json()


def get_embedding(input:List[str]) -> dict:
    completion_payload = dict(input=input)
    response = requests.post(
        f"{BASE_URI}/embedding",
        headers=headers,
        json=completion_payload)
    return response.json()


'''
get_completion(prompt='write a poetry about kittens', echo=True)['text']
get_embedding(input=['abc def', 'ghi jkl', 'lnm pqr'])['embedding']
'''

