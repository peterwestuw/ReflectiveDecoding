'''

This file contains functions used to score the generations of
Reflect Decoding

'''

import time
import math
import numpy as np
import torch
import json

import torch
import torch.nn.functional as F

# get the key for a precalculation

def get_key(source, target):
    return '{}'.format(json.dumps({'source':source, 'target':target}))


def cross_entropy_list(sources, targets, model, cache = None, batch=False, calculate=True):
    '''
    Gets a list of CE values, where the ith item is a list of cross-entropies
    for targets[i] with sources[i] as contexts

    targets and sources are lists of lists of tokens (integers)

    model is a language model

    batch is the batch size to break things up into, batch=False means don't
    break things up into batches, do them all in one go.
    
    CACHING:
    
    cache is a dictionary for single source/target pairs
      accessed by cache[get_key(source,target)]
      it has fields source, target, result
    
    calculate decides whether to immediates calculate for batch of input
      sources/targets or just log them as todo in the cache. To efficiently 
      batch, we can first log many todo calculations by calling cross_entropy_list
      multiple times with calculate=False and the same input cache
      Then finally calling it with calculate=True which will then catch up on all
      todo calculations, caching them together efficiently
    
    '''
    
    ###############################
    # This block handles caching of results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all todo
    # calculations to the cache with calculate = False (won't do them yet)
    # then run with calculate=True to work through all cached calculations
    # in efficient batches
    if cache is not None:

        # log calculations we have not done yet
        for source,target in zip(sources, targets):
            if get_key(source, target) not in cache:
                cache[get_key(source, target)] = {'source': source, 'target':target,'result':None}
        
        # if not calculating, return dummy values
        if not calculate:
            return [1.]*len(sources)
        
        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]
        
        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]
            
            cache_results = cross_entropy_list(sources_todo, targets_todo, model, cache=None, batch=batch)
            for source, target, result in zip(sources_todo,targets_todo, cache_results):
                cache[get_key(source, target)]['result'] = result
    
        ## return results for thie example
        results = [cache[get_key(source, target)]['result'] for source,target in zip(sources, targets)]
        return results
    ###############################        
        
        
        
        
    
    
    
    assert(len(sources ) == len(targets))
    n_seqs = len(sources)
    
    torch.cuda.empty_cache()
    device = model.transformer.wte.weight.device

    # if batching, break it up into smaller pieces
    if batch:
        ce_list = []
        
        n_batches = math.ceil(len(sources) / batch)
        
        list_fun = (lambda v: tqdm(list(v))) if cache is not None else list
        
        #for i in tqdm(list(range(n_batches))):
        for i in list(range(n_batches)):
            ce_list += cross_entropy_list(sources[i*batch:(i+1)*batch], targets[i*batch:(i+1)*batch], model, batch=False)
            #sources, targets = sources[batch:], targets[batch:]
        return ce_list 

    # initialize input tensors
    max_len = max([len(s + t) for s,t in zip(sources, targets)])
    input_ids = torch.zeros((n_seqs, max_len)).long() 
    #-100 is the padding token, which is ignored by F.cross_entropy below
    labels = -100*torch.ones((n_seqs, max_len)).long()
    
    # for each source, target pair, set values in the input tensors
    for i, (source, target) in enumerate(zip(sources,targets)):
        s = torch.tensor(source).long()
        t = torch.tensor(target).long()
        input_ids[i,:len(s)] = s
        input_ids[i,len(s):len(s) + len(t)] = t
        # ignore all predictions except in the target span
        labels[i,len(s):len(s) + len(t)] = t
    
    # get logits from the model
    with torch.no_grad():
        input_ids = input_ids.to(device)
        logits = model(input_ids).logits.cpu()[:,:-1].contiguous()
    
    # get cross-entropies given the logits
    logit_shape = logits.shape
    logits = logits.view(-1, logit_shape[-1])
    ce_list = F.cross_entropy(logits, labels[:,1:].contiguous().view(-1), reduction='none')
    ce_list = ce_list.view(n_seqs, max_len -1).sum(dim=1).squeeze().tolist()
    
    # if one element (i.e. len(sources) == 1), nest it into a list. Otherwise, give full list
    # this just handles an idiosyncracy of the .tolist() function
    try:
        len(ce_list)
    except:
        ce_list = [ce_list]
    
    return ce_list

def score_decode_CE(decode, model, contexts, cache = {}, calculate = True, batch = 1):
    '''
    
    Score a decode by its likelihood to generate contexts
    
    '''
    scores = cross_entropy_list([decode]*len(contexts), contexts, model, cache = cache, batch=batch, calculate=calculate)
    return sum(scores)

def remove_duplicates(options, lowercase = False):
    options = list({tuple(v['phrase']):v for v in options}.values())
    
    options_dict = {}
    
    for entry in options:
        key = entry['phrase_text']
        key = key.strip()
        if lowercase:
            key = key.lower()
            
        if key in options_dict:
            if entry['score'] < options_dict[key]['score']:
                options_dict[key] = entry
            else:
                pass
        else:
            options_dict[key] = entry
            
    return list(options_dict.values())
