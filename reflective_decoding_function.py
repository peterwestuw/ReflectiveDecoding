from PoE_utils import sample_batch_PoE
from scoring_utils import score_decode_CE, remove_duplicates
from generation_processing_utils import process_generations
from weight_learning_utils import get_weights_PoE, prune_contexts
from utils import generate_batch, get_seq_entropy

import torch




def reflective_decode(model_forward, model_backward, encoder,
                      lh, src, rh,
                      config):

    ####
    # Step 1: generate contexts for the input
    ####
    
    device_forward = model_forward.transformer.wte.weight.device

    device_backward = model_backward.transformer.wte.weight.device
    
    ### Generate lh and rh contexts
    contexts_rh = generate_batch(model_forward, 
                                 torch.tensor(lh + src + rh).view(1,-1).to(device_forward), 
                                 top_p=config.top_p, 
                                 length=config.len_gen_context, 
                                 n=config.n_context, batch = config.batch_gen, 
                                 disallowed_toks =config.disallowed_toks)
    contexts_lh = generate_batch(model_backward, 
                             torch.tensor((lh + src + rh)[::-1]).view(1,-1).to(device_backward), 
                             top_p=config.top_p, 
                             length=config.len_gen_context, 
                             n=config.n_context, batch = config.batch_gen, 
                             disallowed_toks = config.disallowed_toks)
    # these are generated in reverse and must be turned ``forward''
    contexts_lh = [c[::-1] for c in contexts_lh]


    ### if we aren't allowing exact copies of the src in context, remove such generations
    if not config.allow_copy:
        source_text = encoder.decode(lh + src + rh)
        contexts_lh = [c for c in contexts_lh if not (source_text in encoder.decode(c)) ]
        contexts_rh = [c for c in contexts_rh if not (source_text in encoder.decode(c)) ]



    ####
    # Step 2: Learn weights
    #### 

    ### First, learn weights using all of the lh contexts that maximize src under a PoE
    weights_lh, distr_lh = get_weights_PoE(model_forward, # language model
                                           [context for context in contexts_lh], # contexts
                                           lh + src + rh, # target
                                           n_it = config.n_it, lr = config.lr_weights, # learning params
                                           disallowed_toks = config.disallowed_toks)
    # prune any weights/contexts whose weights are too low (few contexts->more efficient)
    weights_lh, contexts_lh = prune_contexts(weights_lh, contexts_lh, weight_thresh=config.weight_thresh, keepk=config.keepk)




    ### learn weights using all of the rh contexts that maximize src under a PoE
    weights_rh, distr_rh = get_weights_PoE(model_backward, # language model
                                           [context[::-1] for context in contexts_rh], # contexts, in reverse order
                                           (lh + src + rh)[::-1], # target, reversed order for backward LM
                                           n_it = config.n_it, lr = config.lr_weights, # learning params
                                           disallowed_toks = config.disallowed_toks)
    # prune any weights/contexts whose weights are too low (few contexts->more efficient)
    weights_rh, contexts_rh = prune_contexts(weights_rh, contexts_rh, weight_thresh=config.weight_thresh, keepk=config.keepk)

    ####
    # Step 3: generate from Reflective Decoding function (PoE)
    ####


    ### get generation p_vals using entropy clibration
    p_vals = [i/10. for i in range(1,11)]
    p_val_rh = min(p_vals, key = lambda v: abs( get_seq_entropy(distr_rh[:distr_rh.shape[0]-len(rh)], v) - config.target_entropy  ))
    p_val_lh = min(p_vals, key = lambda v: abs( get_seq_entropy(distr_lh[len(lh):], v) - config.target_entropy  ))


    ### get sampling params
    len_gen_decode = len(src) + config.added_len
    batch_lim = int(min([(30./len_gen_decode), 1.]) *  20)


    ### Generate from Reflective Decoding PoE functions
    # first from lh contexts, using forward LM
    decodes_l2r = sample_batch_PoE(model_forward, 
                                         len_gen_decode,
                                         config.n_gen_decode, 
                                         [c + lh for c in contexts_lh], # contexts with static context
                                         weights_lh, 
                                         p_val_lh,
                                         batch_size = max([int(batch_lim/len(contexts_lh)),1]), 
                                         disallowed_toks = config.disallowed_toks)
    # then from rh contexts, using backward LM
    decodes_r2l = sample_batch_PoE(model_backward, 
                                         len_gen_decode,
                                         config.n_gen_decode, 
                                         [(rh + c)[::-1] for c in contexts_rh], # contexts with static context
                                         weights_rh, 
                                         p_val_rh,
                                         batch_size = max([int(batch_lim/len(contexts_rh)),1]), 
                                         disallowed_toks = config.disallowed_toks)
    # r2l are generated backwards, so reverse these (to be forwards)
    decodes_r2l = [d[::-1] for d in decodes_r2l]
    
    
    reflective_decoding_dict = {'weights_lh':weights_lh,'contexts_lh':contexts_lh,
                               'weights_rh':weights_rh,'contexts_rh':contexts_rh,
                               'p_val_rh':p_val_rh,'p_val_lh':p_val_lh}
    
    return decodes_l2r, decodes_r2l, reflective_decoding_dict