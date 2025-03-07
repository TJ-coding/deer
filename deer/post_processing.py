from beartype import beartype
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

def compute_target_tail_indecies(triplets: List[Tuple[str, str, str]], tail_entitiy_ids: List[str])->List[int]:
    entity_id2index = {entity_id: index for index, entity_id in enumerate(tail_entitiy_ids)}
    target_tail_indecies: List[int] = [entity_id2index[triplets[i][2]] for i in range(len(triplets))]
    return target_tail_indecies

@beartype
def find_ranked_tails(batched_hr_embedding: torch.Tensor, tail_embeddings: torch.Tensor)->torch.Tensor:
    '''Find the ranked tails given the head-relation embedding and the tail embeddings'''
    # Compute cosine similarity between hr_embedding and tail_embeddings
        
    normalised_batched_hr_embeddings = batched_hr_embedding/batched_hr_embedding.norm(dim=1)[:, None]
    normalised_tail_embeddings = tail_embeddings/tail_embeddings.norm(dim=1)[:, None]
    similarities = torch.mm(normalised_batched_hr_embeddings, normalised_tail_embeddings.T)
    
    # Sort the similarities
    sorted_similarities, indecies = torch.sort(similarities, descending=True, dim=1)
    return indecies

@beartype
def compute_target_tail_ranks(hr_embeddings: torch.Tensor,
                              tail_embeddings: torch.Tensor,
                              target_tail_indecies: List[int])->List[int]:
    '''Predict the target tail ranks given the tsvs, target_tail_indecies, entity_id2text, relation_id2text, tail_embeddings, model_inputs, and model_name'''
    tail_ranks: List[int] = []
    number_of_triplets = hr_embeddings.shape[0]
    batch_size: int = 500
    batched_hr_embeddings = torch.split(hr_embeddings, batch_size)
    batched_target_tail_indecies = torch.split(torch.tensor(target_tail_indecies), batch_size)
    
    for batched_hr_embedding, batched_target_tail_indecie in tqdm(zip(batched_hr_embeddings, batched_target_tail_indecies), total=number_of_triplets//batch_size):
        batched_entity_ranks = find_ranked_tails(batched_hr_embedding, tail_embeddings)
        # Get Tail Ranks
        # The line above is really slow, optimize it in the future
        target_tail_ranks = [batched_entity_ranks[i].tolist().index(batched_target_tail_indecie[i]) for i in range(batched_entity_ranks.shape[0])]
        tail_ranks += target_tail_ranks
    return tail_ranks

@beartype
def compute_metrics(tail_ranks: List[int], ks: List[int] = [1, 5, 10])->Dict[str, float]:
    '''Compute the mean rank and the hit@10'''
    results: Dict[str, float] = {}
    results['Mean Rank'] = sum(tail_ranks)/len(tail_ranks)
    results['Mean Reciprical Rank'] = sum([1/rank for rank in tail_ranks])/len(tail_ranks)
    for k in ks:
        results[f'Hit@{k}'] = sum([1 for rank in tail_ranks if rank <= k])/len(tail_ranks)
    return results