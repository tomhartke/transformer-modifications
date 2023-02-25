# Learned query-dependent attention window (hard and soft attention) 

## Implementation
For both hard or soft attention, you just have to replace the CausalSelfAttention class in the nanoGPT 
repository (https://github.com/karpathy/nanoGPT.git).

### Hard attention window results
This approach ends up with similar final validation loss. 
Early layers have heads that generally learn or search long context (full size), 
while final layer heads are mostly shorter range.

One of the goals of hard attention was to reduce the effective context size and therefore
speed up the computation (since the n^2 scaling of computation with context size in transformers is harsh).
Then I could potentially extend the actual context size to be longer 
(things run out of memory around 1064 characters block_size for now).

This reduction in computation ended up being more difficult than expected to implement in pytorch, 
however, since each query has a different hard attention window size (which changes with every sample). 
Therefore the resulting tensor can't be easily compressed (it's ragged across the batches and tokens). 
I didn't pursue this further because the validation loss seemed to not improve with this approach 
(regardless of computational efficiency).

### Soft attention window results
For soft attention, the validation loss ends up the same roughly.
Loss gets down to 1.4571 after 1500 iterations, but then starts overfitting. 

Some heads end up focusing on long range attention (but not all the time).
Some heads focus on shorter range attention (a few tokens).

In all cases here, the attention window is both query and head dependent.