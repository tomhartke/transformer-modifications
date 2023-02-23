# transformer-modifications

## Description
These files are explorations of a toy model transformer implementation. 
I used the nanoGPT transformer by Karpathy (https://github.com/karpathy/nanoGPT.git).

This runs a small transformer on the combined total works of Shakespeare 
(at a character prediction level, without a byte pair encoder). 
The network size is usually 6 layers, 6 heads, with 64 dimensions per head, and a context size (block size) 
of 256 characters.
It produces reasonable sounding text in the style of shakespeare after a few minutes of training. 

File structure: "nanoGPT-basic" folder is just a local copy of the nanoGPT repository.
"Modified-transformers" folder contains my modified code pieces. 

## Goals:
1. Try to improve the transformer architecture by modifying the attention structure or network structure.
2. Generally explore/understand transformers. Vary the network shape (heads, layers, embedding size), 
the dropout, and the context size (block size).

## Transformer alterations:

The approaches tried fall into two categories:
### Query-dependent attention windows 
I project the embedding vector to a single learned parameter (the query window size or attention span) and
   use this to adjust the attention structure from that query to all of it's keys. Specifically, I weigh the attention paid to previous keys based on this context size. 
1. I tried a soft attention window (which just decays the importance further away from the query, with a decay 
      length set by the learned window size for that token).
   > Soft attention: ![Alt text](docs/SoftAttention.jpg?raw=true "Optional")
2. I also tried hard (but still differentiable) attention where I zero out all attention beyond the query size. 
I can still make it differentiable by having the attention decay near the end of the context window.
   > Hard attention: ![Alt text](docs/HardAttention.jpg?raw=true "Optional")

##### Comments 
One goal here was to reduce the computational complexity. In the case of hard attention, one could imagine reducing the attention complexity from n^2 to something closer to linear). 

Another goal was to reduce the "confusion" of the transformer. If the context window is too large, and the position
      embedding is not particularly expressive, then it can probably add entropy to have too large of an attention span.

In hindsight, these query-dependent attention mechanisms are somewhat similar to the routing transformer https://arxiv.org/abs/2003.05997. 

### Attention alteration based on the attention structure
The attention structure for a specific token is calculated: 
>![Alt text](docs/AttentionStructure_a.jpg?raw=true "Optional")
The attention structure includes the relative distance that the token query looks back to 
   find its most important keys. The standard deviation of how far it looks back is also calculated. 
   Sometimes the average key-query logit strength is calculated as well (a proxy for certainty).

These attention alterations are further divided into three methods.
>![Alt text](docs/AttentionStructure_b.jpg?raw=true "Optional")
1. One approach gates the attention head outputs (scales the embedding output) based on an MLP computation 
which sees the full attention structure for a given token (over all heads in a layer).
   1. See: "Scale each head output based on attention structure"
2. A second approach feeds the attention structure for a token into the usual attention MLP (just concatenates it). 
   1. See: "Autoregressive relative location computation". 
3. A third approach generates two new positional embeddings (full size embedding vectors) based off of the attention
structure, and then uses these to gate the elements of the added embedding vector, and as a bias to the embedding.
   1. See: "Change embedding based on based on attention structure"

## Results

Most of the transformer alterations have no effect (detrimental or positive) on the lowest possible
achievable validation loss (around 1.455 or 1.47). 

Here are a few overall takeaways:
1. I think this lack of improvement is because the vector embedding is a sufficiently powerful representation space
to implicitly capture most of the information I am adding explicitly. 
   1. For example, I initially tried to include a separate "importance weight" for each token which would offset the
   calculated attention logit when that token was used as a key, and which would be learned and passed between layers. 
   In this way the network could learn to ignore or emphasize certain tokens. However, this is unnecessary, since the 
   transformer can just place the embedding vector of the key somewhere irrelevant in the high dimensional space, and it
   leads to the same outcome of "ignoring" that information.
2. Another possibility for why I was unable to improve the validation loss is that the shakespeare dataset is too small, and it is not possible to improve the 
validation loss further (of course, fine tuning a larger model can capture higher order reasoning, but 
I am simply training on Shakespeare here).

## How to setup and run the files
Get some GPU hardware. I have been using Jarvislabs (https://jarvislabs.ai/, an online server, with an A100 GPU).
It's simple to start an instance, open VScode, and then clone the necessary repositories locally.

### Explicit steps
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

## Future work
Try sparse attention with multiscale heads looking back further distance.
Could be set up so that total key number for each query is the same for each head (even if some heads 
look back much further). This could be used to compress the computational size of the context window, 
while looking much further back in real distance.

Could try to create something a bit more similar to how humans process ideas.
We have a more sequential process of reasoning, looking at one spot at a time, 
then going back and analyzing some other thing, then summarizing it in our head, 
then coming back and doing something similar in another layer. 

Papers to reference:
Reformer, sparse transformer and strided attention ones.
Different papers that already implemented distance-aware attention
Routing transformer. 
Non-parametric transformer ideas
Memory based and saving information for local review.
Reference my other knowledge graph thing. 

Emphasize the idea that the structure of the attenion is not particularly accessible. 