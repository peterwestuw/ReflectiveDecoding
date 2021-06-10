import os
import logging
import sys

import random
import shutil
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from generation_configs import quora_config, anlg_config

import jsonlines
import json
import sys
import nltk
import numpy as np
from modeling.padded_encoder import Encoder



try:
    shutil.rmtree('outputs/quora')
except:
    print('fail')
    pass
os.mkdir('outputs/quora')



def drop_quotes(s):
    ind_0 = s.find('"')
    ind_1 = s.rfind('"')
    s = s[(ind_0+1):ind_1]
    assert(len(s)) > 0
    return s

encoder = Encoder()

## different bleu cutoffs to try
#
# Note: the cutoff will not be the same as the resulting mean bleu
#
bleu_cutoffs = [1.00, .95, .90, .85, .80, .75, .70, .65, .60, .55, .53, 0.5, 0.45]

def process_reflective(s):
    s = s.replace('\n', ' ')
    s = ' '.join(nltk.tokenize.word_tokenize(s))
    s = s.strip().lower()
    return s


####
# Get data inputs
####

lh = encoder.encode('. "')[1:]
rh = encoder.encode('"') #encoder.encode('?') 

lh_list = []
rh_list = []


with open(quora_config.data_path) as f:
    lines = f.readlines()
    source_texts = [line.strip().split('\t')[0] for line in lines]    
    #source_texts = ['"{}"'.format(s) for s in source_texts]
    target_texts = [line.strip().split('\t')[1] for line in lines]
    ## Add quotation marks to source texts
    source_texts = ['. "{}"'.format(s) for s in source_texts]
    # encode source texts (remove the period, keep the preceding space)
    sources = [encoder.encode(s)[1:] for s in source_texts]
    # take the " off of the beginning and end, as static contexts
    lh_list = [s[0:1] for s in sources]
    rh_list = [s[-1:] for s in sources]
    sources = [s[1:-1] for s in sources]
    targets = [encoder.encode(t) for t in target_texts]
inputs_list = list(zip(sources, targets, lh_list, rh_list))

with open( quora_config.save_path,'r') as f:
    working_dict = json.load(f)
    


src_reflective = []
tgt_reflective = []
dec_reflective = []

reflective_cutoff = {}
for cutoff in bleu_cutoffs:
    reflective_cutoff[cutoff] = []
    
for i in range(len(working_dict)):
    (source, target,_,_) = inputs_list[i]
    phrase = source
    key = str(i)
    if working_dict[key]['result'] is not None:
        lh = working_dict[key]['lh']
        rh = working_dict[key]['rh']
        src_reflective += [ drop_quotes(encoder.decode(lh+ source+ rh)) ]
        tgt_reflective += [encoder.decode(target)]
        
        # don't take any exact copies
        candidate = drop_quotes(encoder.decode(lh+ working_dict[key]['result']['scored_decodes'][0]['phrase'] + rh)).strip()
        for cutoff in bleu_cutoffs:
            candidate_dec = drop_quotes(encoder.decode(lh+ working_dict[key]['result']['scored_decodes'][0]['phrase'] + rh)).strip()
            j = 0

            for opt in working_dict[key]['result']['scored_decodes']:
                candidate_dec = drop_quotes(encoder.decode(lh+ opt['phrase'] + rh)).strip()
                if eval_bleu_sentence(process_reflective(candidate_dec), process_reflective(src_reflective[-1])) < cutoff:
                    break
            reflective_cutoff[cutoff] = reflective_cutoff[cutoff] + [candidate_dec]

        dec_reflective += [candidate]

        
####
# Process generations and such
####
        
src_reflective = list(map(process_reflective, src_reflective))
tgt_reflective = list(map(process_reflective, tgt_reflective))
dec_reflective = list(map(process_reflective, dec_reflective))
for cutoff in reflective_cutoff.keys():
    reflective_cutoff[cutoff] = list(map(process_reflective, reflective_cutoff[cutoff]))

####
# Save with various bleu cutoffs
####

with open('outputs/quora/quora_output_top.txt','w') as f:
    f.writelines(dec_reflective)




for cutoff in reflective_cutoff:

    mean_bleu = round(corpus_bleu([[s] for s in src_reflective[:len(dec_reflective)]],reflective_cutoff[cutoff]),3)
    print(cutoff, mean_bleu)
    
    with open('outputs/quora/quora_output_bleu{}.txt'.format(mean_bleu),'w') as f:
        f.writelines(reflective_cutoff[cutoff])



