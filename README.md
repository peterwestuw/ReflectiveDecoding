# ReflectiveDecoding

This is the repo contains the code for:

This is the official repo for the paper ["Reflective Decoding: Beyond Unidirectional Generation with Off-the-Shelf Language Models"](https://homes.cs.washington.edu/~pawest/ReflectiveDecoding.html) (ACL 2021)

Reflective Decoding is a method with direct applications in paraphrasing and text infilling. 

We include scripts for generating in both of these applications. Reflective Decoding works off the shelf--simply download our models, set up the appropriate packages and you can immediately begin generating.

## Models

Reflective Decoding using forward and backward GPT-2 XL language models, which we train for this work. You can download out pretrained model weights [here](https://github.com/peterwestuw/GPT2ForwardBackward). If you have your own forward/backward LMs you would like to use, you may need to make minimal changes to the code e.g. which model architecture is used--ours differs very slightly from the original GPT-2 but you should be able to substitute in any huggingface model. 

Once you download these models, extract them into the `models/` directory, and point the appropriate model path parameters in `generation_configs.py`. Note that we use 2 GPUs with one model on each. This will depend on what system you are using and can be changed with the `model_device` parameters in `generation_configs.py`.

## Data

To load Quora (paraphrasing) and Abductive Natural Language Generation (infilling) datasets, run `bash get_data.sh` in the `data/` directory. This will download the data in a format that Reflective Decoding can use.

Note, we use subsets of each of these datasets in the original work. If you would like the exact subsets we used, please contact the authors. 

## Code setup

We suggest using conda to setup environment. With conda installed, create a new environment:

```
conda create -n refdec python=3.6
conda activate refdec
```

Next, install [pytorch](https://pytorch.org/). We used version 1.8.1 when testing this code. The exact version will depend on your system (OS, GPUs etc.)

Finally, install the code requirements:


```
pip install -r requirements.txt
```

Once this is done, you should be ready to generate!

## Generating

Note that generation may take 10s of seconds per example. Depending on your system, you can speed this up by setting larger batch sized in `generation_configs.py` or reducing the number of contexts generated (`n_context`) or final generations (`n_gen_decode`) in `generation_configs.py`. This may affect performance, however. The default parameters are the same as in the original work. The inference algorithm relies on random sampling, so the resulting generations will change even with the same parameters.

### Paraphrasing

Once you have installed the appropriate packages, downloaded data and models, and pointed `generation_configs.py` parameters correctly, you can begin generating. 

You can generate paraphrases on the quora corpus by running:

```
python decode.py -version quora -overwrite -verbose -n_ex <n_ex>
```

`-overwite` means the output file is overwritten, and you generate on the given dataset from scratch. If the code halted or stopped unexpectedly and you would like to continue where it left off (i.e. not throw away previous generations/progress) call *without* overwrite

`-verbose` controls whether inputs and outputs are printed during generation. Note, the *best* generation will be printed which may be a direct copy of the source sentence in many cases. As per the original work, a novelty threshold can be imposed to assure paraphrases are different form the source.

`-n_ex` controls how many sentences to paraphrase for. default=1000 meaning the system only paraphrases the first 1000 examples from the input file. 

Note that you can paraphrase for other datasets simply by changing `quora_config.data_path` and  `quora_config.save_path` to point to a new input file (in the save format as the quora data) and name for a new output file.

**Post Processing**: once paraphrases are generated for the whole dataset, you can use the ranked novelty system from the original paper. Simply call ` python cutoff_novelty.py` and system outputs at a variety of novelty levels will be saved in `outputs/quora` with average bleu (1 - novelty) given in the file name.

### Text infilling

This is quite similar to paraphrasing. Simply call:

```
python decode.py -version quora -overwrite -verbose -n_ex <n_ex>
```

As above, you can adapt this to other infilling tasks by changing params in `generation_configs.py`. 
