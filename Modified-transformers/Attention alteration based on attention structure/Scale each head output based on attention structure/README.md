# Scale each head output based on attention structure

### Overall idea:
For each attention head and each query, this self attention block computes meta-information about 
the structure of the resulting attention matrix and then uses an MLP with cross-head knowledge to decide how to 
scale the output value vectors for that token and that head.

Reasoning: Information is missing about the relative location of the resulting attention. Attention just adds
the resulting value vectors linearly. Perhaps keeping this information explicitly might help. For example if 
a specific heads encode different types of information, retaining the explicit relations between the 
attention locations and strengths might be useful. 

### Details:
What is measured:
1. The average logit (weighted by attention). This is roughly how strongly the head is activated.
2. The relative distance from the query to the average key (weighted by attention). 
3. The deviation of the relative distance of the query to the average key location (weighted by attention)

This information is fed to an MLP (a sub-component of the altered attention head) which sees these 3 datapoints for a specific token and for all heads. 
This MLP decides how to scale the output value vector for each head for this token. 

### Updates to make to run it:
Just replace the CausalSelfAttention class.
Also add in the MLP_loc class (keep the old MLP class as well). This just has slightly different format. 

### Outlook:
This doesn't help the ultimate validation loss, but also doesn't hurt. 
The head outputs end up getting scaled by on the order of 2 larger or smaller.
Most likely because the transformer is already sufficiently expressive to gather this information.

Possible reasons:
1. Perhaps the vector-based representation space is already sufficiently large to encode
the effective importance of each head in the output vector.
2. The relative location is somewhat transferred to the next layer in the positional embedding, but this is a bit opaque.