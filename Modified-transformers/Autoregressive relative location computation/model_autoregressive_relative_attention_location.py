class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config

        extra_scale = 4
        self.register_buffer("pos", torch.arange(0, config.block_size, dtype=torch.long) / self.config.block_size)
        self.rel_scale_norm_params = torch.nn.Parameter(data=torch.ones(config.block_size)*self.config.block_size * extra_scale)
        self.rel_bias_norm_params = torch.nn.Parameter(data=torch.zeros(config.block_size))
        self.std_scale_norm_params = torch.nn.Parameter(data=torch.ones(config.block_size)*self.config.block_size * extra_scale)
        self.std_bias_norm_params = torch.nn.Parameter(data=torch.zeros(config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        device = x.device

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Now calculate position of major attention based on softmax
        att_loc_avg = torch.sum(att * self.pos[0:T], -1)
        att_loc_rel = att_loc_avg - self.pos[0:T]
        eps = 1e-12
        att_loc_std = torch.pow(torch.sum(att * ((self.pos[0:T] - att_loc_avg.unsqueeze(-1)) ** 2 + eps), -1), 1.0/2.0)
        # These are shape (B, nh, T)

        def get_auto_regressive_layer_norm_shift_and_scale(input_tensor, scale_params, bias_params):
            # For each token, we want the average and variance over the previous tokens.
            # Then we will manually scale and shift, using parameters based on which token index it is.
            cumsum_denominator = torch.arange(1, T + 1, dtype=torch.long, device=device)
            output_tensor_avg = torch.cumsum(input_tensor, dim=-1) / cumsum_denominator
            input_tensor_shifted = input_tensor - output_tensor_avg

            # the input tensor may have a dimension smaller than the scaling/shifting params
            # if so, only use the params corresponding to the relevant positions.
            final_dim = input_tensor_shifted.shape[-1]
            output_tensor = input_tensor_shifted * scale_params[0:final_dim] + bias_params[0:final_dim]
            return output_tensor

        att_loc_rel = get_auto_regressive_layer_norm_shift_and_scale(att_loc_rel,self.rel_scale_norm_params,
                                                                                 self.rel_bias_norm_params)
        att_loc_std = get_auto_regressive_layer_norm_shift_and_scale(att_loc_std, self.std_scale_norm_params,
                                                                     self.std_bias_norm_params)

        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Get attention location with correct concatenated values for each token.
        att_loc_rel = att_loc_rel.transpose(1, 2).contiguous().view(B, T, self.n_head)
        att_loc_std = att_loc_std.transpose(1, 2).contiguous().view(B, T, self.n_head)
        att_loc = torch.cat((att_loc_rel, att_loc_std), dim=-1) # (B, T, n_head*2)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att_loc

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # increase size of MLP to include importance term as well.
        extra_params_per_head = 2
        other_extra_params = 0
        self.c_fc    = nn.Linear((config.n_embd + config.n_head*extra_params_per_head + other_extra_params),
                                 4 * (config.n_embd + config.n_head*extra_params_per_head + other_extra_params),
                                 bias=config.bias)
        self.c_proj  = nn.Linear(4 * (config.n_embd + config.n_head*extra_params_per_head + other_extra_params),
                                 (config.n_embd),
                                 bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_att_loc = LayerNorm(config.n_head*2, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):

        x_vals, x_att_loc = self.attn(self.ln_1(x))
        x = self.ln_2(x + x_vals)  # shape (b, t, n_emb)
        x_att_loc = self.ln_att_loc(x_att_loc)

        x_and_att_loc = torch.cat((x, x_att_loc), dim=-1)  # shape (b, t, n_emb+n_head*something)
        x = x + self.mlp(x_and_att_loc)
        return x

class GPT(nn.Module):

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        ...
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias') or pn.endswith('norm_params'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                ...
