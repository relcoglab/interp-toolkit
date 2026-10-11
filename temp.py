
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Qwen3-8B
def prompt_qwen(prompt, model_name="Qwen/Qwen1.5-7B-Chat", max_new_tokens=128, temperature=0.7, device=None):
    """
    Loads the Qwen model and generates a response to the given prompt.
    
    Args:
        prompt (str): The input prompt.
        model_name (str): Hugging Face model name.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        device (str or None): Device to use ('cuda', 'cpu', or None for auto).
        
    Returns:
        str: The generated response.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response