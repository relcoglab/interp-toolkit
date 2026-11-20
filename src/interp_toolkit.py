import torch
import transformer_lens
from typing import Optional, List, Union, Tuple
import einops
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.hook_points import HookPoint
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from typing import Literal
import gc

#device should be set once as a global variable and then always used
#we may need make some classes to do this

ROOT_PATH = Path(__file__).resolve().parent

torch.set_grad_enabled(False)

TRANSFORMERLENS_HF_RETRY=1

MODEL_FAMILY = ''

#ENV UTILS

def check_environment():
    """Checks if the required environment variables are set."""
    required_vars = ["HF_HOME", "HF_TOKEN"]
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        # Check the environment variables...
        print(
            f"Hugging face cache inside the dockerfile "
            f"lives at {os.environ.get('HF_HOME')}"        )
        secret_token = os.environ.get("HF_TOKEN")
        redacted = f"{secret_token[:3]}...{secret_token[-3:]}"  
        print(f"Hugging face token is {redacted}")

def load_environment_variables():
    """Loads environment variables from a .env file."""
    load_dotenv()

    if not load_dotenv():
        print("Could not find dotenv file...")
    else:
        print("Successfully loaded dotenv file. ", end="")
        print(f"Environment PROJECT={os.environ.get('PROJECT')}")


#PROMPT UTILS

def load_chat_template(model_name, template_path=None):
    """
    Reads chat_templates.json and returns the template for the given model name.
    """
    if template_path is None:
        template_path = ROOT_PATH /"chat_templates.json"
    template_path = Path(template_path)


    if not template_path.exists():
        raise FileNotFoundError(f"Could not find template file: {template_path}")

    with open(template_path, "r") as f:
        templates = json.load(f)

    if model_name not in templates:
        raise KeyError(
            f"No chat template for '{model_name}'. Available: {list(templates.keys())}"
        )

    return templates[model_name]


def create_prompt(model_family, system_message, user_message):
    """Pulls up the appropriate template for the model family from chat template library stored in 
    chat_templates and formats it with the system and user messages. 
    Suggested use is to loop through stimuli overtop this function."""

    template = load_chat_template(model_family)
    final_prompt = template.format(system_prompt = system_message, user_prompt=user_message)
    #return the result
    return final_prompt


#GPU UTILS

def test_gpus():
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            # Allocate a tensor on each GPU to test if it's functioning properly
            try:
                x = torch.randn((1000, 1000), device=f"cuda:{i}")
                print(f"Successfully created tensor on GPU {i}")
            except Exception as e:
                print(f"Failed to create tensor on GPU {i}: {e}")
    else:
        print("No GPUs are available.")

def clear_gpu_memory():
    #if GPU memory error, clear the GPU cache. If not working, restart the kernel / juypter notebook
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

# DISK UTILS
def delete_hf_model_cache(model_path):
    cache_name = "models--" + model_path.replace("/", "--")
    cache_path = os.path.join(HF_HUB_CACHE, cache_name)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
        print(f"  removed disk cache: {cache_path}")
    else:
        print(f"  no disk cache at: {cache_path}")


#MODEL BASICS

def load_model(
    model_name: str,
    device: str,
    model_family: str,
    checkpoint_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
):
    """The load_model function allows loading in a transformer model onto a specified CUDA device using TransformerLens."""

    # Set the model family as a global variable
    global MODEL_FAMILY
    MODEL_FAMILY = model_family

    try:
        device = torch.device(device)
        

        if checkpoint_path:
            model = TransformerBridge.boot_transformers(
                model_name,
                checkpoint=checkpoint_path,
                device=device,
                dtype=dtype,
                **kwargs
            )
        else:       
            model = TransformerBridge.boot_transformers(
            model_name,
            device=device,
            dtype=dtype,
            **kwargs
        )

        # Move model to device
        model.to(device)
        print(f"Model '{model_name}' loaded successfully on {device}.")

        model.enable_compatibility_mode(no_processing=True)
        model.tokenizer.padding_side = 'left'
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.cfg.device = device # store the device in the model for later use

        
        return model

    except Exception as e:
        raise RuntimeError(f"Problem with loading model '{model_name}': {str(e)}")


def return_model_parameters(model):
    """
    Returns a dictionary of model parameters with their names and shapes.
    
    Args:
        model: The model from which to extract parameters.
        
    Returns:
        A dictionary where keys are parameter names and values are their shapes.
    """
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = param.shape
    return param_dict

def generate(model, input_text: str, max_new_tokens: int = 50) -> str:
    """The generate function takes a model and an input string, and generates text based on the input using the model's generate method."""
    tokens = model.to_tokens(input_text)
    generated_tokens = model.generate(tokens, max_new_tokens=max_new_tokens, device=model.cfg.device)
    generated_text = model.to_string(generated_tokens)
    return generated_text


def get_activations(
    model: transformer_lens.model_bridge.transformer_bridge.TransformerBridge,
    input_text: Union[str, List[str]],
    layer: Optional[int] = None,
    token_pos: Optional[int] = None,
    hook_type: Literal["residual_pre", "residual_post", "mlp", "attn"] = None,
) -> torch.Tensor:
    """Given some model and input, the get_activations function retrieves the hidden layers in the residual stream."""

    valid_hook_types = ["residual_pre", "residual_post", "mlp", "attn"]
    if hook_type not in valid_hook_types:
        raise ValueError(f"hook_type must be one of {valid_hook_types}")

    if layer is not None and (layer < 0 or layer >= model.cfg.n_layers):
        raise ValueError(f"layer must be between 0 and {model.cfg.n_layers-1}")

    # Tokenize input
    if isinstance(input_text, str):
        input_text = [input_text]
    tokens = model.to_tokens(input_text)

    # Run model with cache to collect activations and logits
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False, device=model.cfg.device)

    # Select activation 'hook' type
    if hook_type == "residual_post":
        key_prefix = "blocks"
        key_suffix = "hook_resid_post"

    elif hook_type == "residual_pre":
        key_prefix = "blocks"
        key_suffix = "hook_resid_pre"

    elif hook_type == "mlp":
        key_prefix = "blocks"
        key_suffix = "hook_mlp_out"

    elif hook_type == "attn":
        key_prefix = "blocks"
        key_suffix = "hook_attn_out"

    # extract activations
    if layer is not None:
        # Get specific layer's activations
        key = f"{key_prefix}.{layer}.{key_suffix}"
        activations = cache[key]
            
        if token_pos is not None:
            # get specific token position activations
            activations = activations[:, token_pos, :]  # (batch, seq_len, d_model)
        
    else:
        # Collect activations for all layers
        activations = []
        for l in range(model.cfg.n_layers):
            key = f"{key_prefix}.{l}.{key_suffix}"
            activations.append(cache[key])
        activations = torch.stack(activations, dim=0)  # (n_layers, batch, seq_len, d_model)


        if token_pos is not None:
            # get specific token position activations
            activations = activations[:, :, token_pos, :]

    return activations,logits

def get_pooled_activations(model, 
                           prompt: str,
                            layer: int, 
                            token_pos_indices: range, 
                            hook_type: str):
    """
    Get activations for a given prompt and layer, pooling across specified token positions.
    
    Args:
        model: The model from which to extract activations.
        prompt: The input prompt for the model.
        layer: The layer from which to extract activations.
        token_pos_indices: A list of token position indices to pool over.
        hook_type: The type of hook to use for extracting activations; options: "residual_pre", "residual_post", "mlp", "attn"
        
    Returns:
        A tensor containing the pooled activations across the specified token positions.
    """
    pooled_activations = []
    
    for token_pos in token_pos_indices:
        activations = get_activations(model, prompt, layer=layer, token_pos=token_pos, hook_type=hook_type)
        pooled_activations.append(activations)
    
    # Pooling by taking the mean across the specified token positions
    pooled_activations_tensor = torch.mean(torch.stack(pooled_activations), dim=0)
    
    return pooled_activations_tensor

def extract_activation_vector(model,
                            model_family: str,
                            positive_stimuli: list[str],
                            negative_stimuli: list[str], 
                            token_indices_positive: list[tuple[int]], 
                            token_indices_negative: list[tuple[int]], 
                            hook_type: str,
                            layer: int,
                            system_message: str = '') -> dict:
        
        
    """
    Must pass stimuli and token positions for where to look for activations in the stimuli. 
    To get this across multiple layers, must loop through layers with a higher level function. Same for hook points.
    For each stim item, provide token indices tuple (start and end points) over which to pool activations. Allows a single token position.
    System message is optional--will be added as system prompt to all stimuli in the list.
    """

    #for each item in positive set, get activations, get mean 
    #for each item in negative set, get activations, get mean
    #subtract them

    #loop through all positive stimuli and collect activations
    all_positive_activations = []

    for stim in positive_stimuli:

        #define the prompt and fit into chat template
        this_prompt = create_prompt(model_family, system_message, stim)

        #identify the range of token positions to pool over (could be one token position)
        these_token_inds = range(token_indices_positive[positive_stimuli.index(stim)][0], token_indices_positive[positive_stimuli.index(stim)][1])

        #get the pooled activations across this range of tokens
        this_pooled_activation = get_pooled_activations(model, 
                                                        this_prompt, 
                                                        layer=layer,
                                                          token_pos_indices=these_token_inds,                                                          
                                                          hook_type=hook_type, )

        #add to list of mean activations
        all_positive_activations.append(this_pooled_activation)

    

    mean_positive_activation = torch.stack(all_positive_activations).mean(0)


    #loop through all negative stimuli and collect activations
    all_negative_activations = []
    for stim in negative_stimuli:

        #define the prompt and fit into chat template
        this_prompt = create_prompt(MODEL_FAMILY, system_message, stim)

        #identify the range of token positions to pool over (could be one token position)
        these_token_inds = range(token_indices_negative[negative_stimuli.index(stim)][0], token_indices_negative[negative_stimuli.index(stim)][1])

        #get the pooled activations across this range of tokens
        this_pooled_activation = get_pooled_activations(model, this_prompt, layer=layer, hook_type=hook_type, token_pos=these_token_inds)

        #add to list of mean activations
        all_negative_activations.append(this_pooled_activation)

    mean_negative_activation = torch.stack(all_negative_activations).mean(0)


    direction = mean_positive_activation - mean_negative_activation
    direction = direction / direction.norm()

    del all_negative_activations, all_positive_activations, mean_negative_activation, mean_positive_activation
    gc.collect()     
    torch.cuda.empty_cache()    

    return direction 
