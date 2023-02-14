# Auto-regressive relative location computation 

## Implementation:
This is complicated to implement. You need to replace the Block, MLP, and CausalSelfAttention classes.
Also in the GPT class, we currently have to change configure_optimizers() to have 
"if pn.endswith('bias') or pn.endswith('norm_params'):" to add the shift/scale parameters to those 
without weight decay during training (this lack of weight decay is probably not necessary, 
but the parameters needed to be added into one of the two classes for the GPT configure_optimizers function).

## Comments

This implementation gets the auto-regressive (only looking back in time) avg and std. deviation of the attention location.
Then it learns a scale and bias to apply to this for each token (currently shared across heads, which maybe isn't the best,
but different for each token and layer).
Then it appends this information for all heads to the input to the usual transformer MLP layer. 

Here the ultimate validation loss isn't particularly good (only down to 1.51 validation). Maybe that's something to do 
with the lack of weight decay, or the tying of scaling across heads. 