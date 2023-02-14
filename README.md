# transformer-modifications

## Description
These files are explorations of a toy model transformer implementation 
(the nanoGPT transformer by Karpathy, https://github.com/karpathy/nanoGPT.git).

#### Goals:
1. Try to improve the transformer architecture by modifying the attention structure or network structure.
2. Generally explore/understand transformers. Vary the network shape (heads, layers, embedding size), 
the dropout, and the context size (block size).

## Basic setup steps
1. Clone nanoGPT repository (https://github.com/karpathy/nanoGPT.git)
   1. Check out documentation there if something in nanoGPT is unclear.
2. Install a few things: 
   1. pip install datasets tiktoken wandb tqdm 
3. Prepare dataset: 
   1. python data/shakespeare_char/prepare.py
4. Alter model:
   1. Potentially alter transformer (see "modified-transformers" folder)
4. Train: 
   1. python train.py config/train_shakespeare_char.py --compile=False 
      1. compile=False is to prevent some pytorch bugs.
   2. The training setup can be altered further in the file config>train_shakespeare_char.py. 
   However, I did not find this improved anything significantly (the default settings are reasonable).
7. Sample:
   1. python sample.py --out_dir=out-shakespeare-char --num_samples=1

## Specific transformer alterations