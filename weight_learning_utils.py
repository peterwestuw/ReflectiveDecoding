'''

Functions used to learn the ensemble weights

for PoE in Reflective Decoding

'''

import torch
import torch.nn.functional as F

ce_full = torch.nn.CrossEntropyLoss(reduction='none')
CE = torch.nn.CrossEntropyLoss(reduction='sum')





def get_weights_PoE(model, contexts, target, n_it = 10000, back=False, lr = 0.001, disallowed_toks = []):
    '''
    
    This function gets weights that maximize generating target given an ensemble of contexts
    under the product-of-experts formulation:
    
    argmax w  softmax( π P(target|context)^w_i) 
    
    where P is defined by the model.
    
    
    
    '''             
    device = model.transformer.wte.weight.device

    
    ## first, get distribution over target tokens for each context
    distr_list = []
    logit_list = []
    for i, context in enumerate(contexts):
        toks_c = context
        toks_t = target
        logits = model(torch.tensor(toks_c + toks_t, device = device).view(1,-1))[0].detach()
        for tok in disallowed_toks:
            logits[:,:,tok] = -100000.
        distr_list += [ F.softmax(logits[:,-(len(toks_t) + 1):-1,:], dim=2)]
        ce = ce_full(logits.squeeze()[-(len(toks_t) + 1):-1], torch.tensor(toks_t, device = device)[-len(toks_t):])
        logit_list += [F.log_softmax(logits[:,-(len(toks_t) + 1):-1,:], dim=2)]

    # get a matrix of the logits for P(target|contexts) which will be easy to use
    # for learning weights
    logit_mat = torch.cat(logit_list,dim=0)      
    
    # initialize weights to 0 (uniform)--proto weights are then softmaxed to keep weights
    # a proper distribution
    proto_weights = torch.zeros(len(logit_list), requires_grad=True, device=device)
    
    
    # learn weights that minimize ce of the target
    sm_weight_f = lambda x: F.softmax(x, dim=0)
    opt = torch.optim.Adam([proto_weights], lr = lr)
    for i in range(n_it):
        opt.zero_grad()   
        weights = sm_weight_f(proto_weights)
        logits = (logit_mat*weights.view(-1,1,1)).sum(dim=0)     
        loss = ce_full(logits.squeeze(), torch.tensor(toks_t, device = device)[-len(toks_t):])
        loss.sum().backward()
        opt.step()
        

    # also return distribution over target tokens 
    distr = F.softmax(logits.detach(),dim=1).squeeze()
    
    return weights.cpu().tolist(), distr





def prune_contexts(weights, contexts, weight_thresh = 0., keepk = 10):
    
    zipped = list(zip(weights,contexts))
    zipped.sort(key= lambda v: v[0],reverse=True)
    
    # number of contexts to keep, based on keepk or a threshold for weight magnitude
    keepk = min([keepk, sum([w > weight_thresh for w in weights ])])

    # trim context/weight that do not meet the threshold
    weights, contexts = zip(* zipped[:keepk] ) 
    # renormalize reduced weights (could relearn but this takes longer)
    weights = [w/sum(weights) for w in weights]
    
    return weights, contexts


