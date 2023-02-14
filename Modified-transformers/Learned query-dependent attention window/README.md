# Learned query-dependent attention window (hard and soft attention) 

Updates:
For both hard or soft attention, just have to replace CausalSelfAttention class.



### Hard attention window
Also ends up with similar final validation loss. 
Early heads learn long context, while final heads are mostly shorter. 
They are 

One of the goals of hard attention was the reduce the necessary context size and therefore
speed up the computation (since the n^2 scaling of computation with context size is harsh).
This was more difficult than expected to implement, since each query has a different context size. 
Therefore the resulting tensor can't be easily compressed (it's ragged).

### Soft attention window
For soft attention, loss ends up the same roughly.
Loss gets down to 1.4571 after 1500 iterations, but then starts overfitting. 
The computation is FAST. 

Some heads end up focusing on long range attention (but not all the time).
Some heads focus on shorter range attention (a few tokens).