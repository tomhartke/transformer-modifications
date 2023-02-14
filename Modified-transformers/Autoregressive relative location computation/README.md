# Auto-regressive relative location computation 

This one gets the auto-regressive avg and std. deviation of the atteniton location.
Then it learns a scale and bias to apply to this for each token (currently shared across heads).
Then it appends this information for all heads to the MLP layer. 

This is complicated to implement. Need to replace the Block, MLP, and CausalSelfAttention classes.
Also in the GPT class, we have to change configure_optimizers() to have 
"if pn.endswith('bias') or pn.endswith('norm_params'):" to add the shift/scale parameters to those without weight decay.

Here the ultimate loss isn't particularly good (only down to 1.51 validation.)