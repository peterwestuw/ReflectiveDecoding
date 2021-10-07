'''

General utils for Reflective Decoding

Any functions that don't fit in other files

'''

import torch

# The estimated entropy for sampling a sequence like the source
# we just take this to be the model's distirbutional across samples
# given by the ground-truth contexts
def get_seq_entropy(distr, p_sample = 1., temp = 1.):

    assert(p_sample == 1. or temp == 1.)
    
    if p_sample != 1.:
        return get_seq_entropy_p(distr,p_sample)
    else:
        return get_seq_entropy_temp(distr, temp)
    
def get_seq_entropy_p(distr, p_sample):
    
    values, indices = distr.sort(dim=1, descending = True)
    values = values.cpu()
    indices = indices.cpu()

    distr_cs = values.cumsum(dim=1)
    cutoff_map = distr_cs < p_sample

    # make sure resulting distr covers at least p_sample
    cutoff_map[:,1:] = cutoff_map[:,:-1]
    cutoff_map[:,0] = True

    H = 0

    for i in range(values.shape[0]):
        tmp_dist = values[i, cutoff_map[i]]
        tmp_dist = tmp_dist/(tmp_dist.sum())

        H += -(tmp_dist*(tmp_dist.log())).sum()

    return H

import torch.nn.functional as F
def get_seq_entropy_temp(distr, temp):
    
    logits = distr.log()
    
    logits = logits/temp
    
    distr = F.softmax(logits, dim = 1)

    H = 0

    
    
    for i in range(distr.shape[0]):
        
        H += -((distr[i]*distr[i].log())[distr[i] > 0]).sum()

    return H


def generate_batch(model, inp, top_p=0.8, temperature = 1., length=40, n=40, batch = 50, disallowed_toks = []):
    '''
    
    Just call the model generate function, but allow batching in case we need many generations,
    and automatically convert between lists and tensors (input and output are lists)
    
    '''
    gens = []
    
    # length should be generated length
    length = inp.numel() + length 
    n_left = n # number of generations left to do
    while n_left > 0:
        
        n_batch = min([batch, n_left])
        n_left = n_left - n_batch

        gen_batch = model.generate(inp,
                             do_sample = True,
                             top_k=0, # disable default top_k=50
                             top_p=top_p, 
                             temperature=temperature, 
                             bad_words_ids=[[t] for t in disallowed_toks],
                             min_length = length,
                             max_length = length,
                             num_return_sequences = n_batch)

        # we store sequences as lists 
        # and don't include the input in the sequence
        gens.extend([g[inp.numel():] for g in gen_batch.tolist()])

        torch.cuda.empty_cache()

    return gens

