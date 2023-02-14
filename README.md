# transformer-modifications

## Description
These files are explorations of a toy model transformer implementation. 
I used the nanoGPT transformer by Karpathy (https://github.com/karpathy/nanoGPT.git).

This runs a small transformer on the combined total works of Shakespeare 
(at a character prediction level, without a byte pair encoder). 
The network size is usually 6 layers, 6 heads, with 64 dimensions per head, and a context size (block size) 
of 256 characters.
It produces reasonable sounding text in the style of shakespeare after a few minutes of training. 

#### Goals:
1. Try to improve the transformer architecture by modifying the attention structure or network structure.
2. Generally explore/understand transformers. Vary the network shape (heads, layers, embedding size), 
the dropout, and the context size (block size).

#### Results:

The approaches tried fall into two categories:
1. Query-dependent attention windows 
   1. I project the embedding vector to a single learned parameter (the query window size or attention span) and
   use this to adjust the attention structure from that query to all of it's keys.
   2. Specifically, I weigh the attention paid to previous keys based on this context size. 
      1. I tried a soft attention window (which just decays the importance further away from the query, with a decay 
      length set by the learned window size for that token).
      2. I also tried hard (but still differentiable) attention where I zero out all attention beyond the query size. 
      I can still make it differentiable by having the attention decay near the end of the context window. 
   3. One goal here was to reduce the computational complexity 
      1. In the case of hard attention, one could imagine reducing the attention complexity from n^2 to something closer to linear).
   4. Another goal was to reduce the "confusion" of the transformer. If the context window is too large, and the position
      embedding is not particularly expressive, then it can probably add entropy to have too large of an attention span.
   5. In hindsight, this is somewhat similar to the routing transformer https://arxiv.org/abs/2003.05997.
2. Attention alteration based on the attention structure 
   1. The attention structure for a specific token is: the relative distance that the token query looks back to 
   find its most important keys. The standard deviation of how far it looks back is also calculated. 
   Sometimes the average key-query logit strength is calculated as well (a proxy for certainty).
   2. These attention alterations are further divided into three methods. 
      1. One approach gates the attention head outputs (scales the embedding output) based on an MLP computation 
      which sees the full attention structure for a given token (over all heads in a layer).
         1. See: "Scale each head output based on attention structure"
      2. A second approach feeds the attention structure for a token into the usual attention MLP (just concatenates it). 
         1. See: "Autoregressive relative location computation". 
      3. A third approach generates two new positional embeddings (full size embedding vectors) based off of the attention
      structure, and then uses these to gate the elements of the added embedding vector, and as a bias to the embedding.
         1. See: "Change embedding based on based on attention structure"

Most of the transformer alterations have no effect (detrimental or positive) on the lowest possible
achievable validation loss (around 1.455 or 1.47). More details can be found in subfolders. Here are a few overall takeaways:
1. I think this is because the vector embedding is a sufficiently powerful representation space
to implicitly capture most of the information I am adding explicitly. 
   1. For example, I initially tried to include a separate "importance weight" for each token which would offset the
   calculated attention logit when that token was used as a key, and which would be learned and passed between layers. 
   In this way the network could learn to ignore or emphasize certain tokens. However, this is unnecessary, since the 
   transformer can just place the embedding vector of the key somewhere irrelevant in the high dimensional space, and it
   leads to the same outcome of "ignoring" that information.
2. Another possibility for why I was unable is that the shakespeare dataset is too small, and it is not possible to improve the 
validation loss further (of course, fine tuning a larger model can capture higher order reasoning, but 
I am simply training on Shakespeare here).

## How to setup and run the files
Get some GPU hardware. I have been using Jarvislabs (https://jarvislabs.ai/, an online server, with an A100 GPU).
It's simple to start an instance, open VScode, and then clone the necessary repositories locally.

#### Explicit steps:
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