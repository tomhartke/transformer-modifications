# Change embedding based on attention structure (shift and scale)

For this, you have to update the Block, and CausalSelfAttention classes. 
You also have to add the MLP_loc class, which is a slightly altered MLP layer. 

This one is a bit more complicated.

It computes the attention positional information for all the heads for a given token.
Then it feeds that into an MLP to generate vectors of size n_emb.
One vector gates the original data from the attention layer in the transformer.
The other vector adds a bias to the MLP input. 

Final result doesn't help. 