from typing import Dict, List, Tuple
from unittest import TestCase, main
from deer.encoder_model import PromptEOL_Encoder
from deer.prompt_templates import query2prompts, tail_entities2prompts
from deer.post_processing import compute_target_tail_indecies, compute_target_tail_ranks, compute_metrics
from deer.utils import knowledge_probe


class TestKnowledgeProbing(TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
    
    def test_all(self):
        triplets: List[Tuple[str, str, str]] = [('subject 1', 'predicate 1', 'object 1'), ('subject 2', 'predicate 2', 'object 2')]
        entity_id2text: Dict[str, str] = {'subject 1': 'entity 1', 'subject 2': 'entity 2', 'object 1': 'entity 3', 'object 2': 'entity 4'}
        relation_id2text: Dict[str, str] = {'predicate 1': 'relation 1', 'predicate 2': 'relation 2'}
        fewshot_prompt = '(subject 0, predicate 0, object 0)'
        encoder = PromptEOL_Encoder('facebook/opt-125m', cuda=False)
        entity_id2definition = {'subject 1': 'definition 1', 'subject 2': 'definition 2', 'object 1': 'definition 3', 'object 2': 'definition 4'}
        
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
        print(results)
        
    def test_utils(self):
        triplets: List[Tuple[str, str, str]] = [('subject 1', 'predicate 1', 'object 1'), ('subject 2', 'predicate 2', 'object 2')]
        entity_id2text: Dict[str, str] = {'subject 1': 'entity 1', 'subject 2': 'entity 2', 'object 1': 'entity 3', 'object 2': 'entity 4'}
        relation_id2text: Dict[str, str] = {'predicate 1': 'relation 1', 'predicate 2': 'relation 2'}
        fewshot_prompt = '(subject 0, predicate 0, object 0)'
        entity_id2definition = {'subject 1': 'definition 1', 'subject 2': 'definition 2', 'object 1': 'definition 3', 'object 2': 'definition 4'}
        
        results: dict[str, float] = knowledge_probe(triplets, entity_id2text, relation_id2text, entity_id2definition, fewshot_prompt, 'facebook/opt-125m', False)
        print(results)
        
        
if __name__ == '__main__':
    main()
    