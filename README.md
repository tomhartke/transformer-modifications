# transformer-modifications

## Description
These files are explorations of a toy model transformer implementation.
I used the nanoGPT transformer by Karpathy (https://github.com/karpathy/nanoGPT.git). 
This repo creates a small transformer in pyTorch which is trained on the combined total works of Shakespeare 
(at a character prediction level, without a byte pair encoder), producing reasonable sounding text in the correct style after a few minutes of training. 

The purpose of these files is mostly as a personal record, and learning tool.




## Goals and results:
1. Try to improve the transformer architecture by modifying the attention structure or network structure (no luck, unsurprisingly).
2. Generally explore/understand transformers. Vary the network shape (heads, layers, embedding size), 
the dropout, and the context size (block size).

#### File structure
* "nanoGPT-basic" folder is just a local copy of the nanoGPT repository by Karpathy (https://github.com/karpathy/nanoGPT.git).
  * I did not write any of this! (I hope it is ok to just copy here for reference).
* "Modified-transformers" folder contains my modified code pieces. 

## Transformer alterations:

The base transformer implementation I altered has a network size of usually 6 layers, 6 heads, with 64 dimensions per head, and a context size (block size) 
of 256 characters. There are usually around 10 million parameters, and few minutes of training on a GPU.

The approaches I took can be split into two categories:
### 1. Query-dependent attention windows 
The first approach I took is to project the embedding vector to a single learned parameter (the query window size or attention span) and
   use this to adjust the attention structure from that query to all of it's keys. Specifically, I weigh the attention paid to previous keys based on this context size. 
1. I tried a soft attention window (which just decays the importance further away from the query, with a decay 
      length set by the learned window size for that token).
   > Soft attention: ![Alt text](docs/SoftAttention.jpg?raw=true "Optional")
2. I also tried hard (but still differentiable) attention where I zero out all attention beyond the query size. 
I can still make it differentiable by having the attention decay near the end of the context window.
   > Hard attention: ![Alt text](docs/HardAttention.jpg?raw=true "Optional")

##### Comments 
* One goal here was to reduce the computational complexity. 
  * In the case of hard attention, one could imagine reducing the attention complexity from n^2 
  to something closer to linear. Most attention heads learn a very short context length, 
  and a few learn longer context lengths.
* Another goal was to reduce the "confusion" of the transformer. If the context window is too large, and the position
      embedding is not particularly expressive, then it can probably add entropy to have too large of an attention span.
* Useful references:
  * In hindsight, these query-dependent attention mechanisms are somewhat similar to the routing 
  transformer https://arxiv.org/abs/2003.05997.
  * I found this blog post on transformer extensions very useful in hindsight: 
  https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
    * It includes discussion of methods of distance aware attention that have been explored in the literature,
    including adding a linear offset to the biases (although that seems to be query-independent) 
    and learning a head-dependent context size (similar to my hard attention, but again independent of query). 

### 2. Attention alteration based on the attention structure
The second approach I took is to try to preserve the structure of the attention (ie the spatial distribution of
words that are being paid attention to). 
* It seems like the linear superposition of value vectors that are added together
at the attention output are unaware of the distribution of where they came from, though preserving this information may be useful.
  * If there were no positional embeddings, then this information would strictly be lost.
* This might be false (ie the information is indeed still there) if the original positional embeddings of the transformer 
are sufficiently robust and interpretable as the information propagates through the network.
  * I haven't tested this explicitly (try to see how positional embeddings propagate and are preserved through the layers),
  but it seems likely to me that the later layers can't very clearly interpret where information is coming from.

Here is an outline of this approach:
First, the attention structure for a specific token is calculated: 
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
achievable validation loss (around 1.455 or 1.47 on the shakespeare character dataset). 

Here are a few overall takeaways:
1. I think this lack of improvement is because the vector embedding is a sufficiently powerful representation space
to implicitly capture most of the information I am adding explicitly. 
   1. For example, I initially tried to include a separate "importance weight" for each token which would offset the
   calculated attention logit when that token was used as a key, and which would be learned and passed between layers. 
   In this way the network could learn to ignore or emphasize certain tokens. However, this is unnecessary, since the 
   transformer can just place the embedding vector of the key somewhere irrelevant in the high dimensional embedding space, and it
   leads to the same outcome of "ignoring" that information.
2. Another possibility for why I was unable to improve the validation loss is that the shakespeare dataset is too small, and it is not possible to improve the 
validation loss further 
(As an example of this, one can imagine that fine tuning a larger model can capture higher order reasoning, but 
I am simply training on Shakespeare here).

## How to setup and run the files
Get some GPU hardware. I have been using Jarvislabs (https://jarvislabs.ai/, an online server, with an A100 GPU).
It's simple to start an instance, open VScode, and then clone the necessary repositories locally.

### Explicit steps
1. Clone nanoGPT repository (https://github.com/karpathy/nanoGPT.git)
   1. Check out documentation there if something in nanoGPT is unclear.
2. Install a few things: 
   ```
   pip install datasets tiktoken wandb tqdm 
   ```
3. Prepare dataset: 
   ```
   python data/shakespeare_char/prepare.py
   ```
4. Alter model:
   1. Potentially alter transformer (see "modified-transformers" folder, and grab and copy chunks of code as necessary).
4. Train: 
   ```
   python train.py config/train_shakespeare_char.py --compile=False 
   ```
      1. compile=False is to prevent some pytorch bugs.
   2. The training setup can be altered further in the file config>train_shakespeare_char.py. 
   you can change the layer number, embedding size, head number, learning rates, etc.
   However, I did not find this improved anything significantly (the default settings are reasonable).
7. Sample:
   ```
   python sample.py --out_dir=out-shakespeare-char --num_samples=1
   ```

## Future work
Many more people with better experience have attacked this problem for longer, so I'll probably stop here for now, until I know more.

One idea is to try sparse attention with multiscale heads looking back further distance, but with the same
total number of tokens attended to by each query. 
That is, some heads look back every 16 tokens, some every 4, some every 2, etc. 
This could be used to compress the computational size of the context window, 
while looking much further back in real distance.

## References 

Here are a set of references to look through and be aware of relating to transformer extensions (I haven't read all of them in detail, and this is certainly not exhaustive):
* https://arxiv.org/abs/1911.05507 ("Compressive Transformers for Long-Range Sequence Modelling")
* https://arxiv.org/abs/1911.00172 ("Generalization through Memorization: Nearest Neighbor Language Models")
* https://arxiv.org/abs/2102.02557 ("Adaptive Semiparametric Language Models")
* https://arxiv.org/abs/2203.08913 ("Memorizing Transformers")
* https://arxiv.org/abs/2108.12409 ("Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation")
* https://aclanthology.org/2021.naacl-main.166/ ("DA-Transformer: Distance-aware Transformer")
* https://arxiv.org/abs/1807.03819 ("Universal Transformers")
* https://arxiv.org/abs/1802.05751 ("Image Transformer")
* https://arxiv.org/abs/1904.10509 ("Generating Long Sequences with Sparse Transformers")
* https://arxiv.org/abs/1911.02972 ("Blockwise Self-Attention for Long Document Understanding")
* https://arxiv.org/abs/2001.04451 ("Reformer: The Efficient Transformer")
* https://arxiv.org/abs/2006.04768 ("Linformer: Self-Attention with Linear Complexity")
* https://arxiv.org/abs/1803.02155 ("Self-Attention with Relative Position Representations")
* https://arxiv.org/abs/1901.02860 ("Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context")
* https://arxiv.org/abs/2104.09864 ("RoFormer: Enhanced Transformer with Rotary Position Embedding")
* https://arxiv.org/abs/2009.14794 ("Rethinking Attention with Performers")
* https://arxiv.org/abs/2011.04006 ("Long Range Arena: A Benchmark for Efficient Transformers")

Good blog posts:
* https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
* https://chengh.medium.com/evolution-of-fast-and-efficient-transformers-ec0378257994 
