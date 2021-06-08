'''

processing of generations prior to scoring


'''

import regex as re

regex_split = lambda sent: [s for s in  re.split('\W\s|\?|:',sent ) if len(s) > 0]

def flat_split(sentences, fun):
    out = []
    for s in sentences:
        out += fun(s)
    return out

split_newline = lambda v: v.split('\n')
split_eot = lambda v: v.split('<|endoftext|>')
split_colon = lambda v: v.split(':')

import regex as re
regex_split = lambda sent: [s for s in  re.split('\W\s|\?|"|:',sent ) if len(s) > 0]

# first tokenize along \n and eot, then sentence tokenize the pieces
hyper_blast = lambda s: flat_split(flat_split(flat_split( [s], split_eot), regex_split) , nltk.tokenize.sent_tokenize)

regex_sentence_split = lambda s: re.findall('[?.!\"#()]+|[a-zA-Z0-9_,\' ]{2,}',s)


def get_decode_options_sentences(encoder, toks, gen_dir, only_first = False):
    '''
    This function sentence-tokenizes the given sentence,
    then for each consecutive list of sentences starting 
    from the first (gen_dir='l2r') or last (gen_dir='r2l') sentence, returns
    the input tokens corresponding to those sentences
    
    Only first means we only take the first generated sentence, by sentence
    tokenization (this still depends on direction. If gen_dir = 'r2l' then the
    'first' generated sentence is actually the rightmost one)
    '''


    # sequence corresponding to the input toks
    s = encoder.decode(toks )

    # list of sentences in sequence s (text)
    sent_list = regex_sentence_split(s)

    # 
    if gen_dir == 'r2l':
        sent_list = list(reversed(sent_list))

    # only use the first generated sentence
    if only_first:
        sent_list = sent_list[0:1]
        
    # this will contain all possible sentences beginning with the first
    sentece_sequences = []

    for num_sent in range(len(sent_list)):
        subsent = []
        for i in range(len(toks)):

            if gen_dir == 'l2r':
                subsent = toks[:i+1]
            elif gen_dir == 'r2l':
                subsent = list(reversed(list(reversed(toks)) [:i+1]))
            else:
                assert(False)

            # if this sequence of tokens contains all of the desired sentences, break
            if all( [ sent_list[j] in encoder.decode(subsent) for j in range(num_sent + 1) ]):
                break
        encoder.decode(subsent)
        sentece_sequences += [subsent]
    return sentece_sequences



def process_generations(encoder, decodes_l2r, decodes_r2l, src):


    decodes_all = []
    
    ## first, generate options form raw generations with sentence tokenization
    for decode in decodes_l2r:
        decodes_all += get_decode_options_sentences(encoder, decode, 'l2r', only_first = False)
    for decode in decodes_r2l:
        decodes_all += get_decode_options_sentences(encoder, decode, 'r2l', only_first = False)


    ## Only allow well-formed sentences
    # if the source contained " allow it, otherwise do not
    if '"' in encoder.decode(src):
        regex_filter='[a-zA-Z0-9_, ]*[a-zA-Z0-9_,]+.*[a-zA-Z0-9_,]+'
    else:
        regex_filter='[a-zA-Z0-9_, ]*[a-zA-Z0-9_,]+[^"]*[a-zA-Z0-9_,]+'

    # remove any exact duplicates
    candidate_phrases = [list(c) for c in  set([tuple(c) for c in decodes_all])]

    if not(regex_filter is None):
        candidate_phrases = list(filter( lambda x: re.fullmatch(regex_filter ,encoder.decode(x)) is not None,  candidate_phrases) )

    return decodes_all