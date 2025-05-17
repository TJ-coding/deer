'''
Contains code for extracting the last hidden vector from a transformer model.
'''

from typing import List, TypedDict
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from beartype import beartype

class LLMInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
    
class PromptEOL_Encoder:
    '''
    A class to encode prompts using a transformer model.
    '''
    @beartype
    def __init__(self, model_name:str, cuda:bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if cuda==False:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device: str = 'cpu'
            print('Using CPU')
        else:
            number_of_devices = torch.cuda.device_count()
            print(f'Number of devices: {number_of_devices}')
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",  # Distributes layers across GPUs
            offload_folder=None )
            self.device = 'cuda'
            
        self.model.eval()
        
    def __call__(self, prompts: List[str])->torch.Tensor:
        llm_inputs: List[LLMInputs] = PromptEOL_Encoder.text2llm_inputs(self.tokenizer, prompts)
        encodings: torch.Tensor = PromptEOL_Encoder.encode(llm_inputs, self.model, self.device)
        
        return encodings
        
    @beartype
    @staticmethod
    def text2llm_inputs(tokenizer: PreTrainedTokenizerBase, prompts: List[str])->List[LLMInputs]:
        llm_inputs = []
        for prompt in tqdm(prompts, total=len(prompts)):
            llm_inputs.append(tokenizer(prompt, return_tensors="pt"))
        return llm_inputs
        
    @beartype
    @staticmethod
    def encode(llm_inputs: List[LLMInputs], model: PreTrainedModel, device: torch.device)->torch.Tensor:
        last_vectors: List[torch.Tensor] = []
        with torch.no_grad():
            for llm_input in tqdm(llm_inputs, total=len(llm_inputs)):
                llm_input = {k: v.to(device) for k, v in llm_input.items()}
                outputs = model(**llm_input, output_hidden_states=True)

                # Get the last hidden state vector
                last_hidden_state = outputs.hidden_states[-1]  # Last layer's hidden states

                # Extract the vector corresponding to the last token
                last_vector = last_hidden_state[:, -1, :]
                if device!=torch.device('cpu'):
                    last_vector = last_vector.cpu()
                last_vectors.append(last_vector)
        last_vectors = torch.concat(last_vectors, dim=0)
        return last_vectors
