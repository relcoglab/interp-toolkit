import torch
import transformer_lens
from typing import Optional, List, Union, Tuple
import einops
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.hook_points import HookPoint

#device should be set once as a global variable and then always used
#we may need make some classes to do this

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



def get_hidden(
    model: transformer_lens.model_bridge.transformer_bridge.TransformerBridge,
    input_text: Union[str, List[str]],
    layer: Optional[int] = None,
    return_type: str = "residual"
) -> torch.Tensor:
    """Given some model and input, the get_hidden function retrieves the hidden layers in the residual stream."""

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
        
    else:
        # Collect activations for all layers
        activations = []
        for l in range(model.cfg.n_layers):
            key = f"{key_prefix}.{l}.{key_suffix}"
            activations.append(cache[key])
        activations = torch.stack(activations, dim=0)  # (n_layers, batch, seq_len, d_model)

    return activations,logits