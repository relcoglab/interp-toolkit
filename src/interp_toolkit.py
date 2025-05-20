import torch
from typing import Optional, List, Union, Tuple
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
import einops

def load_model(
    model_name: str,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> HookedTransformer:
    """The load_model function allows some transformer model using TransformerLens."""

    try:
        # Set the device for the model
        if device is None:
            device = get_device()
        device = torch.device(device)

        # Load model; checkpoint or pre-trained
        if checkpoint_path:
            model = HookedTransformer.from_pretrained(
                model_name,
                checkpoint=checkpoint_path,
                device=device,
                **kwargs
            )
        else:
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                **kwargs
            )

        # Move model to device
        model.to(device)
        print(f"Model '{model_name}' loaded successfully on {device}.")
        
        return model

    except Exception as e:
        raise RuntimeError(f"Problem with loading model '{model_name}': {str(e)}")


def get_hidden(
    model: HookedTransformer,
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

    # Run model with cache to collect activations
    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

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

    return activations