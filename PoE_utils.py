'''

PoE sampling


This file contains functions used to generate under a 
product-of-experts over different contexts

MAIN: see sample_batch_PoE 

this is the main function used by other methods

'''



import torch.nn.functional as F
import torch
from torch import nn
import sys

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0 and top_p < 1.: 
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        
        logits[indices_to_remove] = filter_value

        
    
    return logits

def sample_sequence_PoE(model, length, contexts, num_samples=1, top_k=0, top_p=0.0, temp=1., device='cpu', context_weights = None, disallowed_toks = [], include_uncond = False):
    
    logsm = nn.LogSoftmax(dim=-1)
    # this will interpolate with the unconditional distribution. Not implemented yet...
    if include_uncond:
        assert(False)
        
        
    torch.cuda.empty_cache()
    

    disallowed_bias = None
    
    # all contexts should have the same length
    assert( all([len(context) == len(contexts[0]) for context in contexts ]))
    
    contexts = [torch.tensor(context, dtype=torch.long, device=device) for context in contexts]
#    contexts = [context.unsqueeze(0).repeat(num_samples, 1) for context in contexts]


    # inner list is for a specific generation (i.e. generated[i] is for generaiton i, 
    # generated[i][j] is the jth context for generation i)
    generated = [[context for context in contexts] for _ in range(num_samples)] 
    
    past = None
    
    input_tensor = torch.zeros((len(contexts)*num_samples, length + contexts[0].numel() )).long().to(device)

    len_context_last = 0
    
    if context_weights is None: # if no weights, weight equally
        context_weights = [1. for _ in contexts]
        print('NO CONTEXT WEIGHTS in generation')
              
    with torch.no_grad():
        for i_len in range(length):
            try:

                end_tok_list = [gen[0].numel() for gen in generated]
                max_len = max(end_tok_list)

                ind = 0
                for gen in generated:
                    for context in gen:
                        input_tensor[ind, :len(context)] = context
                        len_context = len(context)
                        ind += 1

                inp = {'input_ids': input_tensor[:,len_context_last:len_context], 'past_key_values':past}
                len_context_last = len_context
                
                output = model(**inp)
                past = output[1]
        
                ## mask disallowed tokens from all logits
                # initialize disallowed bias if not already
                if disallowed_bias is None:
                    disallowed_bias = torch.zeros_like(output[0][0,0])

                    for tok in disallowed_toks:
                        disallowed_bias[tok] = -1000000.
                    disallowed_bias.view(1,1,-1)
                        
                # weight logits down for disallowed tokens (making p effectively 0)
                output = (output[0] + disallowed_bias, output[1])
            
        
                outputs = [output[0][i:i+1] for i in range(output[0].shape[0])]

            
                if context_weights is None: # if no weights, weight equally
                    context_weights = [1. for _ in contexts]
                #else:
                #    context_weights = [1.*s/sum(context_weights) for s in context_weights]

                ind = 0
                for i, gen in enumerate(generated):
                    # output probabilities all start as 0, then we do weighted adding
                    out = torch.zeros((outputs[0][0, -1, :].numel()), dtype=torch.float, device=device)

                    
                    # add weighted log softmax
                    for context, weight in zip(gen, context_weights):
                        out += (weight*F.log_softmax(  outputs[ind][0, -1, :] , dim=-1) )
                        
                        
                        ind += 1
                    
                    out = F.softmax(out, dim = -1)
                   

                    next_token_logits = torch.log(out)
                    
                    # apply temperature if using it
                    if temp!=1.:
                        next_token_logits = next_token_logits/temp

                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)


                    # append new token to all contexts
                    generated[i] = [torch.cat((g, next_token), dim=0) for g in gen]

            except:
                print('FAILED at len = {}, tensor_size = {}'.format(i_len, input_tensor.shape) )
                print("Unexpected error:", sys.exc_info())
                assert(False)
    
    return [gen[0] for gen in generated]


# this version uses batched sampling
def sample_batch_PoE(model, len_gen, n_gen, contexts, context_weights, top_p,batch_size = 1, disallowed_toks = []):
    torch.cuda.empty_cache()
    device = model.transformer.wte.weight.device
    
    decode_list = []
    
    out_tokens_list = []
    n_left = n_gen
    while n_left > 0:
        n_samples = min([n_left, batch_size])
        n_left = n_left - n_samples
        
        samples = sample_sequence_PoE(model, len_gen, contexts, top_p=top_p, device = device, context_weights=context_weights, num_samples = n_samples, disallowed_toks = disallowed_toks)
        samples = [sample.tolist()[len(contexts[0]):] for sample in samples]
        
        out_tokens_list += samples
    
    for out_tokens in out_tokens_list:
        decode_list += [out_tokens]
        
    torch.cuda.empty_cache()
    return(decode_list)