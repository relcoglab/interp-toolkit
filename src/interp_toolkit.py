from xml.parsers.expat import model

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
import numpy as np
import re
import warnings
from contextlib import suppress



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

    try:
        template = load_chat_template(model_family)
    except Exception as e:
        print(f"Error loading chat template: {e}")
        return None

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

#PLOT UTILS

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show(renderer)

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

## ACTIVATION UTILS

def get_activations(
    model: transformer_lens.model_bridge.transformer_bridge.TransformerBridge,
    input_text: Union[str, List[str]], 
    token_pos: Union[int, tuple],
    hook_type_filter: str = None,
    stop_at_layer: int = None) -> torch.Tensor:

    
    """Given some model and input,
      the get_activations function retrieves the hidden layers in the residual stream.
      
      model: TransformerBidge model object
      input_text: the (fully formatted) input prompt for the model or list of prompts for batch processing
      hook_type_filter: name filter to specify the type of hook to use for extracting activations; 
      stop_at_layer: the layer at which to stop extracting activations (if None, returns all)
      token_pos: the token position from which to extract activations (if none returns all)
      
      returns  cache, logits:
        cache: a dict containing tensors, keyed by name         
        logits: a tensor containing the logits for the specified layer and token position

        This basically returns any number of hook points simultaneously. 
        To get tensors out of the cache, must filter by name and turn into a list of tensors.
        Use filter_activation_cache() to return tensors for singular hook points.
      """

    #tokenize the prompt input
    #tokens must bea list of tokenized inputs, one for each prompt in the batch
    if isinstance(input_text, str):
        input_text = [input_text]
    tokens = model.to_tokens(input_text) 


    # run model with cache to collect activations and logits
    # using the names filter saves memory so we add it directly here
    # # same with token positions. note we can pass a tuple to get a slice 

    if(hook_type_filter is None):
            hook_type_filter = ".*"  # Match all if no filter is provided

    #there is an annoying user warning generated when hook types are filtered that we want to ignore
    with warnings.catch_warnings():

        warnings.filterwarnings('ignore', category=UserWarning)

        logits, cache = model.run_with_cache(tokens,
                                          remove_batch_dim=False, 
                                          device=model.cfg.device, 
                                          pos_slice=token_pos,
                                          stop_at_layer=stop_at_layer,
                                          names_filter = lambda s: bool(re.search(hook_type_filter, s)))


    return cache,logits

def filter_activation_cache(cache, model,hook_name: str, layer=None):


    """ Takes an activation cache (a dictionary object) and returns a list of tensors
    Provide the requested hook name (must be unique) and a cache object.
    Returns activations: a list of tensors indexed by layer.

    This can be useful for organizing activation caches by layer into a tensor.

    Activation cache dimensions:(n_layers, batch, num tokens, d_model)

    Check that there is a valid input layer given the # layers of the model passed 
    if layer is not None and (layer < 0 or layer >= model.cfg.n_layers):
        raise ValueError(f"layer must be between 0 and {model.cfg.n_layers-1}") """

    activations = []

    key_prefix = 'blocks'


    if layer is not None:
        if layer < 0 or layer >= model.cfg.n_layers:
            raise ValueError(f"layer must be between 0 and {model.cfg.n_layers-1}")
        key = f"{key_prefix}.{layer}.{hook_name}"
        if key in cache:
            layer_cache = cache[key]
            these_dims = layer_cache.shape
            activations.append(layer_cache)
        else:
            print('Error: key not found in cache. Returning a tensor of zeros.')
            layer_cache = torch.zeros((1, 1, model.cfg.d_model), device=model.cfg.device) #if the key is not found, return a tensor of zeros with the correct shape
            activations.append(layer_cache)
    else:
        layer_cache = []
        
        for l in range(model.cfg.n_layers):

            key = f"{key_prefix}.{l}.{hook_name}"

            if key in cache:
                layer_cache = cache[key]  
                these_dims = layer_cache.shape
                
            elif layer_cache != []:
                print('Error: key not found in cache. Returning a tensor of zeros.')
                layer_cache = torch.zeros((layer_cache.shape), device=model.cfg.device) #if the key is not found, return a tensor of zeros with the correct shape
            
            activations.append(layer_cache)        
        

    print(f"Returning tensor with dimensions: {len(activations)} layers, {these_dims[0]} prompt batch(es), {these_dims[1]} token(s), {these_dims[2]} model dimensions")
    activations = torch.stack(activations, dim=0)  


    return activations

def get_pooled_activations(model, 
                           prompt: str,
                           token_pos_indices: tuple[int], 
                           hook_name: str):
    """
    Get activations for a given prompt and layer, pooling across specified token positions.
    
    Args:
        model: The model from which to extract activations.
        prompt: The (fully formatted) input prompt for the model.
        layer: The layer from which to extract activations.
        token_pos_indices: a range of token position indices to pool over.
        hook_name: The exact name of the hook to use for extracting activations.
        
    Returns:
        A tensor containing the pooled activations across the specified token positions.
    """
    print(token_pos_indices)

    #pass a single or a batch of prompts, a range of token indices, and 
    activation_cache, _ = get_activations(model = model, 
                                           input_text = prompt,
                                           token_pos=token_pos_indices,
                                           hook_type_filter=hook_name)

    
    activation_tensor = filter_activation_cache(activation_cache, model, hook_name=hook_name)
    # print(f"Activation tensor shape: {activation_tensor.shape} (layers, batch, tokens, d_model)")
    
    # Pooling by taking the mean across the token position dimension
    pooled_activations_tensor = torch.mean(activation_tensor, dim=2) 
    these_dims = pooled_activations_tensor.shape
    print(f"Returning tensor with dimensions:  {these_dims[0]} layers, {these_dims[1]} prompt batch(es), {these_dims[2]} model dimensions")
       
    
    return pooled_activations_tensor

### FUNCTIONALITY FOR MEANS DIFFERENCE VECTOR EXTRACTION AND PROJECTION

def extract_activation_vector(model,
                            model_family: str,
                            positive_stimuli: list[str],
                            negative_stimuli: list[str], 
                            token_indices_positive: list[tuple[int]], 
                            token_indices_negative: list[tuple[int]], 
                            hook_name: str,
                            system_message: str = '') -> dict:
        
        
    """
    Defines an activation vector using a means-difference approach.
    Produces a mean activation vector for a positive set of stimuli and a mean activation vector for a negative set of stimuli,
    and then subtracts the two to produce a direction vector. 
    Provide list of positive and negative stimuli and token positions from which to extract the activations.


    positive_stimuli: the list of positive stimuli.
    negative_stimuli: the list of negative stimuli.
    token_indices_positive: the list of token positions for the positive stimuli.
    token_indices_negative: the list of token positions for the negative stimuli.
    hook_name: the exact name of hook to use for extracting activations.
    layer: the layer from which to extract activations.
    system_message: an optional system message to include in all stimuli.

    To get this across multiple layers, loop through layers with a higher level function and aggregate.
    """


    all_positive_activations = []


    for stim in positive_stimuli:

        #define the prompt and fit into chat template
        this_prompt = create_prompt(model_family, system_message, stim)

        #identify the range of token positions to pool over (could be one token position)
        these_token_inds = (token_indices_positive[positive_stimuli.index(stim)][0], token_indices_positive[positive_stimuli.index(stim)][1])
  
        #get the pooled activations across this range of tokens
        this_pooled_activation = get_pooled_activations(model = model, 
                                                        prompt=this_prompt, 
                                                        token_pos_indices=these_token_inds,                                                          
                                                        hook_name=hook_name)

        #add to list of mean activations
        all_positive_activations.append(this_pooled_activation[:,0,:]) #remove the batch dimension, which is 1 because we are only passing one prompt at a time
    
    #create average of the positive activations across all positive stimuli
    #this has shape: n_stim, n-layers, d_model. We want to average across the n_stim dimension, which is 0
    mean_positive_activation = torch.stack(all_positive_activations).mean(0)
    


    #loop through all negative stimuli and collect activations
    all_negative_activations = []
    for stim in negative_stimuli:

        #define the prompt and fit into chat template
        this_prompt = create_prompt(model_family, system_message, stim)

        #identify the range of token positions to pool over (could be one token position)
        these_token_inds = (token_indices_negative[negative_stimuli.index(stim)][0], token_indices_negative[negative_stimuli.index(stim)][1])

        #get the pooled activations across this range of tokens
        this_pooled_activation = get_pooled_activations(model = model, 
                                                        prompt=this_prompt, 
                                                        token_pos_indices=these_token_inds,                                                          
                                                        hook_name=hook_name) #add to list of mean activations
        
        all_negative_activations.append(this_pooled_activation[:,0,:]) #remove the batch dimension, which is 1 because we are only passing one prompt at a time

    #create average of the negative activations across all negative stimuli
    #this has shape: n_stim, n-layers, d_model. We want to average across the n_stim dimension, which is 0
    mean_negative_activation = torch.stack(all_negative_activations).mean(0)

    #calculate the direction vector by subtracting the mean negative activation from the mean positive activation
    direction = mean_positive_activation - mean_negative_activation
    direction = direction / direction.norm()

    assert direction.shape[-1] == model.cfg.d_model, f"Error: Direction vector shape {direction.shape} does not match model dimension {model.cfg.d_model}"
    print(f"Returning tensor with dimensions:  {direction.shape[0]} layers,  {direction.shape[1]} model dimensions")
        
    del all_negative_activations, all_positive_activations, mean_negative_activation, mean_positive_activation
    gc.collect()     
    torch.cuda.empty_cache()    

    return direction 

def project_stimuli_onto_direction(stimuli: list[str],
                                model: transformer_lens.model_bridge.transformer_bridge.TransformerBridge,
                                activation_direction: torch.Tensor,
                                hook_name: str,
                                token_ind: Union[int, tuple],
                                system_message: str = '',
                                layer: int = None):

    """     Project a list of stimuli onto a specific activation direction.

    #Gets activation to each stimulus in list (defaut: at last token)
    #Projects that activation using inner product to activation direction vector
    # If no layer is specified, it will return ALL layers 
    # Returns projections, which has dims (n stimuli) if a specific layer is indicated,
    # otherwise has dims nstimuli, nlayers.

    #stimuli: list of stimuli to project onto the activation direction.
    #activation_direction: the activation direction vector to project onto.
    #hook_name: exact hook name to use for extracting activations.
    #token_ind: the token position or tuple of indices from which to extract activations;
    #system_message: an optional system message to include in all stimuli.

    #note that this is weird because then you are projecting from each layer of the stimuli to one
    #specific direction presumably derived from a specific layer.
    #This use isn't recommended. """
    
    assert activation_direction.shape[-1] == model.cfg.d_model,f"Error: Activation direction shape {activation_direction.shape} does not match model dimension {model.cfg.d_model}. Must pass a 1D vector with size = d-model."


    #batch the stimuli 
    prompts_batch = [] 
    for stim in stimuli:        
        this_prompt = create_prompt(MODEL_FAMILY, system_message, stim)
        prompts_batch.append(this_prompt)


    #get activations for all of the stimuli in all layers
    activation_cache, _ = get_activations(
                        model = model,
                        input_text = prompts_batch,
                        token_pos= token_ind,
                        hook_type_filter = hook_name)
    
    #these_activations has dims [n_layers, batch, tokens, d_model]
    # print(activation_cache['blocks.0.hook_' + hook_name].shape)

    #create tensor with dimensions [n_layers, batch, tokens, d_model] 
    activation_tensor = filter_activation_cache(activation_cache, model, hook_name=hook_name)


    #collapse across the token dimension, leaving only the layer and d_model dimensions
    activation_tensor = torch.mean(activation_tensor, dim=(2)) 
    #normalizes the activations to unit length; applies to last dimension, which is d_model, thus the activations
    # activation_norm = activation_tensor / activation_tensor.norm(dim=-1, keepdim=True)    

    #move to cpu because otherwise there is an error in numpy;it needs things on cpu for some reason
    activation_tensor = activation_tensor.float()
    activation_norm= activation_tensor / (
    activation_tensor.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)) #copilot wrote this last part.
   
    activation_norm = activation_norm.detach().cpu().numpy()

    activation_direction = activation_direction.float().detach().cpu().numpy()

    assert activation_norm.ndim == 3, f"Error: Activation tensor has {activation_norm.ndim} dimensions; expected 3 dimensions (n_layers, batch, d_model)."

    #if the activation direction is provided for all layers, then project layerwise
    if activation_direction.ndim > 1 and activation_direction.shape[0] == model.cfg.n_layers:

        # project each layer's activations onto the corresponding activation direction
        for layer_idx in range(model.cfg.n_layers):

            layer_activations = activation_norm[layer_idx,:,:]
            layer_direction = activation_direction[layer_idx,:]

            layer_projections = np.inner(layer_activations, layer_direction)
            
            if layer_idx == 0:
                projections = layer_projections
            else:
                projections = np.vstack((projections, layer_projections))   

    elif activation_direction.ndim == 1:
        # project activations onto the single activation direction provided       
        projections = np.inner(activation_norm,activation_direction)


    print(f"Returning projections with dimensions: {projections.shape[0]} layers, {projections.shape[1]} prompt batch(es)")
    gc.collect()
    torch.cuda.empty_cache()

    return projections
      
