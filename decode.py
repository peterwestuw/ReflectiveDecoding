'''

This script runs Reflective Decoding on text infilling (``anlg'')
or paraphrasing (``quora'')

To make changes to settings, see generation_configs.py


Unless ``overwrite'' is set to True, running this will continue generation
from the last time this script was called. 

DO NOT change settings without calling
overwrite, or different outputs in the same output file
may be generated using different settings

Control version with:
    -version quora  (paraphrasing)
    -version anlg (text infilling)

'''


from generation_configs import quora_config, anlg_config
import json 
from reflective_decoding_function import reflective_decode
from tqdm import tqdm
import torch
from modeling.modeling_opengpt2 import OpenGPT2LMHeadModel
from modeling.padded_encoder import Encoder
from generation_processing_utils import process_generations
from scoring_utils import score_decode_CE, remove_duplicates
import os
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=str, default = '') # either 'anlg' or 'quora'
    parser.add_argument('-n_ex', type=int, default = 1000) # do inference on the first n_ex in the input file
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-verbose', action='store_true')
    
    opt = parser.parse_args()
    
    
    assert(opt.version in ['quora','anlg'])
    
    
    
    #####
    # Get config and encoder
    #####
    ## first, get the correct config
    if opt.version  == 'quora':
        config = quora_config
    elif opt.version  == 'anlg':
        config = anlg_config
    encoder = Encoder()


    #####
    # Load data and any past generation results
    #####
    ### first, get inputs
    inputs = []
    ## Then, load the inputs
    if opt.version  == 'quora':
        with open(config.data_path) as f:
            # only do the first opt.n_ex examples
            lines = f.readlines()[:opt.n_ex]
            source_texts = [line.strip().split('\t')[0] for line in lines]    
            ## Add quotation marks to source texts (this helps with question paraphrasing)
            source_texts = ['. "{}"'.format(s) for s in source_texts]
            # encode source texts (remove the period, keep the preceding space)
            sources = [encoder.encode(s)[1:] for s in source_texts]
            # take the " off of the beginning and end, as static contexts
            lh_list = [s[0:1] for s in sources]
            rh_list = [s[-1:] for s in sources]
            sources = [s[1:-1] for s in sources]
            inputs = zip(lh_list, sources, rh_list)

    elif opt.version  == 'anlg':
        with open(anlg_config.data_path) as f:
            # only do the first opt.n_ex
            for line in f.readlines()[:opt.n_ex]:
                d = json.loads(line)

                lh = encoder.encode(' {}'.format(d["obs1"])) # lh is obs 1
                src = [] # src is empty -- we are filling this in
                rh = encoder.encode(' {}'.format(d["obs2"])) # rh is obs 2

                inputs += [(lh,src,rh)]


    ####
    # handle dictionary for saving results
    ####
    
    # if we haven't created output dicitonary, create it
    if not os.path.isfile(config.save_path) or opt.overwrite :
        print('='*50)
        print('No output dict yet (or overwrite == True). Writing blank dict to {}'.format(config.save_path))
        print('='*50)
        out_dict = {str(inp):{'lh':list(inp[0]), 'src':list(inp[1]),'rh':list(inp[2]), 'result': None} for inp in inputs }
        with open(config.save_path,'w') as f:
            f.write(json.dumps(out_dict))
    # load output dict (may alread have outputs)   
    print('='*50)
    print('Loading output dict from {}...'.format(config.save_path))
    print('WARNING: if you have changed configs, remove this file and rerun to start from scratch or run with --overwrite')
    print('='*50)
    with open(config.save_path) as f:
        out_dict = json.loads(f.read())
        print('='*50)
        print('{} out of {} examples already done'.format(len([d for d in out_dict.values() if d['result'] is not None]),len(out_dict)))
        print('='*50)  
        
        
        
        
    #####
    # Load models
    #####

    print('='*50)
    print('Loading forward and backward language models...'.format(config.save_path))
    device_forward = torch.device(config.forward_model_device)
    device_backward = torch.device(config.backward_model_device)
    model_forward = OpenGPT2LMHeadModel.from_pretrained(config.forward_model_path).eval().to(device_forward)
    model_backward = OpenGPT2LMHeadModel.from_pretrained(config.backward_model_path).eval().to(device_backward)
    print('done')
    print('='*50)
    
    
    
    
    
    
    ############################
    ############################
    
    ####
    # Generation loop over all inputs
    ####

    for key in tqdm(out_dict.keys()):
        # skip examples we have already done
        if out_dict[key]['result'] is not None:
            continue
            
        # get inputs for this examples
        lh, src, rh = out_dict[key]['lh'],out_dict[key]['src'], out_dict[key]['rh']
        #####
        # First, generate with reflective decoding
        #####
        decodes_l2r, decodes_r2l, rd_dict = reflective_decode(model_forward, model_backward, encoder,lh, src, rh, config)
        contexts_lh, contexts_rh = rd_dict['contexts_lh'], rd_dict['contexts_rh']

        #####
        # Second, process and score the generations
        #####

        ## process raw generations to get formatted and sentence-tokenized options
        decodes_all = process_generations(encoder, decodes_l2r,decodes_r2l, src )
        
        ### Get scores (using caching and batching for efficient scoring)
        cache_l2r = {}
        cache_r2l = {}
        scored_decodes = []
        # first, log all calculations to do for batches CE (calculate = False)
        for decode in decodes_all:
            ### score differently for paraphrasing and infilling
            # For paraphrasing: score based on ability to go from decode
            #                   to generated contexts   
            if config.scoring == 'paraphrase':
                score_decode_CE(lh + decode + rh, model_forward, contexts_rh, cache = cache_l2r, calculate = False, batch = 1)
                score_decode_CE((lh + decode + rh)[::-1], model_backward, [c[::-1] for c in contexts_lh], cache = cache_r2l, calculate = False, batch = 1)
            # For Infilling: score based on ability to go from decode to
            #                o1 and o2 (i.e. lh, rh)
            elif config.scoring == 'infill':
                score_decode_CE(lh + decode, model_forward, [rh], cache = cache_l2r, calculate = False, batch = 1)
                score_decode_CE((decode + rh)[::-1], model_backward, [lh[::-1]], cache = cache_r2l, calculate = False, batch = 1)
        # do batched calculation (calculate = True)
        for decode in decodes_all:
            if config.scoring == 'paraphrase':    
                score_rh = score_decode_CE(lh + decode + rh, model_forward, contexts_rh, cache = cache_l2r, calculate = True, batch = config.batch_score)
                score_lh = score_decode_CE((lh + decode + rh)[::-1], model_backward, [c[::-1] for c in contexts_lh], cache = cache_r2l, calculate = True, batch = config.batch_score)
            elif config.scoring == 'infill':
                score_rh = score_decode_CE(lh + decode, model_forward, [rh], cache = cache_l2r, calculate = True, batch = config.batch_score)
                score_lh = score_decode_CE((decode + rh)[::-1], model_backward, [lh[::-1]], cache = cache_r2l, calculate = True, batch = config.batch_score)
            scored_decodes.append({'phrase':decode, 'phrase_text':encoder.decode(decode), 'score':score_lh + score_rh,
                                  'score_lh': score_lh, 'score_rh':score_rh})

        ### For aNLG, we remove anything that doesn't make both o1 and o2 (lh and rh) more likely
        if config.scoring == 'infill':
            score_rh_null = score_decode_CE((lh), model_forward, [rh], cache = cache_l2r, calculate = True, batch = config.batch_score)
            score_lh_null = score_decode_CE((rh)[::-1], model_backward, [lh[::-1]], cache = cache_r2l, calculate = True, batch = config.batch_score)
            scored_decodes = [d for d in scored_decodes if (d['score_lh'] < score_lh_null) and (d['score_rh'] < score_rh_null)]

        ### remove any duplicate generations  
        scored_decodes = remove_duplicates(scored_decodes, lowercase=False)
        scored_decodes.sort(key=lambda v: v['score'])


        #####
        # Third, save the results for this example
        #####
        # define results dictionary
        result = {'scored_decodes':scored_decodes, # scored outputs
                 'reflective_decode_dict': rd_dict} # other results from rd
        # resave outdict with the results
        out_dict[key]['result'] = result
        with open(config.save_path,'w') as f:
            f.write(json.dumps(out_dict))
        # if verbose, print example 
        if opt.verbose:
            print('='*50)
            print('Input\nlh:{}||src:{}||rh:{}'.format(encoder.decode(lh),
                                                      encoder.decode(src),
                                                      encoder.decode(rh)))
            print('Best generation:{}'.format(scored_decodes[0]['phrase_text']))
            print('='*50)
    

    
    
if __name__ == '__main__':
    main()