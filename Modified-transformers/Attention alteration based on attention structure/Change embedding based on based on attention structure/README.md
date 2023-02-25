# Change embedding based on attention structure (shift and scale)

## Implementation:
For this, you have to update the Block, and CausalSelfAttention classes in the nanoGPT repository 
(https://github.com/karpathy/nanoGPT.git). 
You also have to add the MLP_loc class, which is a slightly altered MLP layer. 

## Comments:

This implementation is a bit more complicated. 
* It computes the attention positional information for all the heads for a given token.
* Then it feeds that into an MLP to generate vectors of size n_emb.
* One vector gates the original data from this attention layer before it gets fed into the MLP.
* The other vector adds a bias to the MLP input. 

Final result is that this doesn't help.