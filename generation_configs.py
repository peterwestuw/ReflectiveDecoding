'''

Configuration for Quora Paraphrasing

'''

quora_config = lambda x: None

## context generation params
quora_config.top_p = 0.7
quora_config.n_context = 80
quora_config.batch_gen = 40
quora_config.disallowed_toks = [198, 199, 629]
quora_config.len_gen_context = 50
quora_config.allow_copy = False

## PoE weight learning params
quora_config.n_it = 160
quora_config.lr_weights = 0.1
quora_config.weight_thresh = 0.05
quora_config.keepk = 6

## Reflective Decoding generation params
quora_config.target_entropy = 4.
quora_config.n_gen_decode = 30
quora_config.added_len = 5 # length to generate beyond length or original source

## processing and scoring params
quora_config.batch_score = 15
quora_config.scoring = 'paraphrase'

## data loading and saving
quora_config.data_path = '/home/pawest/data/quora_questions/split/test.txt'
quora_config.save_path = 'outputs/quora.json'



'''

Configuration for aNLG infilling

'''


anlg_config = lambda x: None

## context generation params
anlg_config.top_p = 0.9
anlg_config.n_context = 100
anlg_config.batch_gen = 50
anlg_config.disallowed_toks = [198, 199, 629]
anlg_config.len_gen_context = 30
anlg_config.allow_copy = False

## PoE weight learning params
anlg_config.n_it = 160
anlg_config.lr_weights = 0.1
anlg_config.weight_thresh = 0.05
anlg_config.keepk = 8

## Reflective Decoding generation params
anlg_config.target_entropy = 6.
anlg_config.n_gen_decode = 30
anlg_config.added_len = 20 # length to generate beyond length or original source

## processing and scoring params
anlg_config.batch_score = 15
anlg_config.scoring = 'infill'

## data loading and saving
anlg_config.data_path = 'data/anlg/dev.jsonl'
anlg_config.save_path = 'outputs/anlg.json'



'''

Shared params

'''

anlg_config.forward_model_path = quora_config.forward_model_path = '/home/pawest/grover_models/opengpt2_pytorch_forward/'
anlg_config.backward_model_path = quora_config.backward_model_path = '/home/pawest/grover_models/opengpt2_pytorch_backward/'
anlg_config.forward_model_device = quora_config.forward_model_device = 'cuda:0'
anlg_config.backward_model_device = quora_config.backward_model_device = 'cuda:1'
