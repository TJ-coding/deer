from typing import Dict, List, Tuple, Optional
import torch
from beartype import beartype
import os

from .encoder_model import PromptEOL_Encoder
from .prompt_templates import query2prompts, tail_entities2prompts
from .post_processing import compute_target_tail_indecies, compute_target_tail_ranks, compute_metrics

@beartype
def save_encodings(embeddings: torch.Tensor, embeddings_save_path: str):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    concatenated_embeddings = embeddings  # Already a tensor
    torch.save(concatenated_embeddings, embeddings_save_path, pickle_protocol=4)
    print(f"Saved embeddings to {embeddings_save_path} with shape {concatenated_embeddings.shape}")

@beartype
def load_encodings(embeddings_load_path: str) -> torch.Tensor:
    if not os.path.exists(embeddings_load_path):
        raise FileNotFoundError(f"File not found: {embeddings_load_path}")
    loaded = torch.load(embeddings_load_path)
    print(f"Loaded embeddings from {embeddings_load_path} with shape {loaded.shape}")
    return loaded

def knowledge_probe(
    triplets: List[Tuple[str, str, str]],
    entity_id2text: Dict[str, str],
    relation_id2text: Dict[str, str],
    entity_id2definition: Dict[str, str],
    fewshot_prompt: str,
    model_name: str,
    cuda: bool,
    query_encodings_load_path: Optional[str] = None,
    tail_encodings_load_path: Optional[str] = None,
    query_encodings_save_path: Optional[str] = None,
    tail_encodings_save_path: Optional[str] = None
) -> Dict[str, float]:
    
    encoder = PromptEOL_Encoder(model_name, cuda=cuda)

    # Query encodings
    if query_encodings_load_path:
        query_encodings = load_encodings(query_encodings_load_path)
    else:
        query_prompts = query2prompts(
            triplets,
            entity_id2text,
            relation_id2text,
            fewshot_prompt=fewshot_prompt,
            entity_id2definition=entity_id2definition
        )
        query_encodings = encoder(query_prompts)
        if query_encodings_save_path:
            save_encodings(query_encodings, query_encodings_save_path)

    # Tail encodings
    if tail_encodings_load_path:
        tail_encodings = load_encodings(tail_encodings_load_path)
    else:
        tail_prompts = tail_entities2prompts(
            list(entity_id2text.values()),
            list(entity_id2definition.values())
        )
        tail_encodings = encoder(tail_prompts)
        if tail_encodings_save_path:
            save_encodings(tail_encodings, tail_encodings_save_path)

    target_tail_indecies: List[int] = compute_target_tail_indecies(triplets, list(entity_id2text.keys()))
    target_tail_ranks: List[int] = compute_target_tail_ranks(query_encodings, tail_encodings, target_tail_indecies)
    results: Dict[str, float] = compute_metrics(target_tail_ranks)

    return results
