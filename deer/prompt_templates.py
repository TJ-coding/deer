'''
Contains code for creating prompts
'''

from typing import Dict, List, Optional, Tuple
from beartype import beartype
from tqdm import tqdm

@beartype
def query2prompts(triplets: List[Tuple[str, str, str]],
                    entity_id2text: Dict[str, str], 
                    relation_id2text: Dict[str, str], 
                    fewshot_prompt: Optional[str] = False,
                    entity_id2definition: Optional[Dict[str, str]] = None
                              )->List[str]:
    head_entity_ids, relation_ids, tail_entity_ids = zip(*triplets)
    prompts: List[str] = []
    if entity_id2definition is not None:
        print('Using definitions')
    for i in range(len(triplets)):
        head_entity = entity_id2text[head_entity_ids[i]]
        relation = relation_id2text[relation_ids[i]]
        text = f'''\n({head_entity}, {relation}, '''
        if entity_id2definition is not None:
            definition = entity_id2definition[head_entity_ids[i]]
            text = f'''{head_entity} - {definition}\n({head_entity}, {relation}, '''
        if fewshot_prompt:
            text = f'''{fewshot_prompt}\n{text}'''
        prompts.append(text)
    return prompts

def tail_entities2prompts(entities: List[str], definitions: Optional[List[str]])->List[str]:
    prompts: List[str] = []
    for i, entity in tqdm(enumerate(entities), total=len(entities)):
        
        prompt_sent = f"This sentence: \"{entity}\" means in one word: \""
        if definitions is not None:
            prompt_sent = f"{entity} - {definitions[i]}"+'\nThis sentence: "{word}" means in one word: "{one word}"'+f"\nThis sentence: \"{entity}\" means in one word: \""
        prompts.append(prompt_sent)
    return prompts