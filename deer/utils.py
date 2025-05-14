from typing import Dict, List, Tuple
import torch
from beartype import beartype

from .encoder_model import PromptEOL_Encoder
from .prompt_templates import query2prompts, tail_entities2prompts
from .post_processing import compute_target_tail_indecies, compute_target_tail_ranks, compute_metrics

@beartype
def save_encodings(embeddings:List[torch.Tensor], embeddings_save_path:str):
    '''
    Save the embeddings to a file.
    
    Args:
    embeddings: List of torch.Tensor
    embeddings_save_path: str
    '''
    concatenated_embeddings = torch.concat(embeddings, dim=0)
    concatenated_embeddings = concatenated_embeddings.numpy()
    print('Shape of concatenated embeddings')
    print(concatenated_embeddings.shape)
    torch.save(concatenated_embeddings, embeddings_save_path, pickle_protocol=4)
    
def knowledge_probe(triplets: List[Tuple[str, str, str]], entity_id2text: Dict[str, str],
                    relation_id2text:Dict[str, str], entity_id2definition:Dict[str, str], fewshot_prompt:str, 
                    model_name: str, cuda: bool )->Dict[str, float]:
    encoder = PromptEOL_Encoder(model_name, cuda=cuda)    
    query_prompts = query2prompts(triplets,
                entity_id2text,
                relation_id2text, 
                fewshot_prompt = fewshot_prompt,
                entity_id2definition = entity_id2definition)
    query_encodings = encoder(query_prompts)
    
    tail_prompts = tail_entities2prompts(list(entity_id2text.values()), list(entity_id2definition.values())) 
    tail_encodings = encoder(tail_prompts)
    
    target_tail_indecies:List[int] = compute_target_tail_indecies(triplets, list(entity_id2text.keys()))
    target_tail_ranks: List[int] = compute_target_tail_ranks(query_encodings, tail_encodings, target_tail_indecies)
    results: dict[str, float] = compute_metrics(target_tail_ranks)
    return results
