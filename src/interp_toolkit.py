import torch
import transformer_lens
from typing import Optional, List, Union, Tuple
import einops
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.hook_points import HookPoint
from pathlib import Path
import json
import os

#device should be set once as a global variable and then always used
#we may need make some classes to do this

ROOT_PATH = Path(__file__).resolve().parent


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
    checkpoint_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
):
    """The load_model function allows loading in a transformer model onto a specified CUDA device using TransformerLens."""

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
    generated_tokens = model.generate(tokens, max_new_tokens=max_new_tokens)
    generated_text = model.to_string(generated_tokens)
    return generated_text


def get_activations(
    model: transformer_lens.model_bridge.transformer_bridge.TransformerBridge,
    input_text: Union[str, List[str]],
    layer: Optional[int] = None,
    token_pos: Optional[int] = None,
    return_type: str = "residual"
) -> torch.Tensor:
    """Given some model and input, the get_activations function retrieves the hidden layers in the residual stream."""

    valid_return_types = ("residual", "mlp", "attn")

    if return_type not in valid_return_types:
        raise ValueError(f"return_type must be one of {valid_return_types}")

    if layer is not None and (layer < 0 or layer >= model.cfg.n_layers):
        raise ValueError(f"layer must be between 0 and {model.cfg.n_layers-1}")

    # Tokenize input
    if isinstance(input_text, str):
        input_text = [input_text]
    tokens = model.to_tokens(input_text)

    # Run model with cache to collect activations and logits
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False, device=model.cfg.device)

    # Select activation type
    if return_type == "residual":
        key_prefix = "blocks"
        key_suffix = "hook_resid_post"

    elif return_type == "mlp":
        key_prefix = "blocks"
        key_suffix = "hook_mlp_out"

    elif return_type == "attn":
        key_prefix = "blocks"
        key_suffix = "hook_attn_out"

    # Extract activations
    if layer is not None:
        # Get specific layer's activations
        key = f"{key_prefix}.{layer}.{key_suffix}"
        activations = cache[key]
        
    if token_pos is not None:
        # Get specific token's activations
        activations = activations[:, :, token_pos, :]
        
    else:
        # Collect activations for all layers
        activations = []
        for l in range(model.cfg.n_layers):
            key = f"{key_prefix}.{l}.{key_suffix}"
            activations.append(cache[key])
        activations = torch.stack(activations, dim=0)  # (n_layers, batch, seq_len, d_model)

    return activations,logits

