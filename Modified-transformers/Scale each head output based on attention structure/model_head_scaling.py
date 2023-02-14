class MLP_loc(nn.Module):

    def __init__(self, config):
        super().__init__()
        n_params_att_loc = (config.n_head*3)
        expansion_factor_att_loc = 4
        self.c_fc    = nn.Linear(n_params_att_loc,
                                 expansion_factor_att_loc * n_params_att_loc,
                                 bias=config.bias)
        self.c_proj  = nn.Linear(expansion_factor_att_loc * n_params_att_loc,
                                 config.n_head,
                                 bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

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

        self.register_buffer("pos", torch.arange(0, config.block_size, dtype=torch.long) / self.config.block_size)
        self.mlp_scale = MLP_loc(config)
        # self.mlp_bias = MLP_loc(config)
        self.mlp_scale_sigmoid = nn.Sigmoid()
        self.ln_importance = LayerNorm(config.n_head, bias=config.bias)
        self.ln_rel = LayerNorm(config.n_head, bias=config.bias)
        self.ln_std = LayerNorm(config.n_head, bias=config.bias)

        self.print_num = 0

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        device = x.device

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_logits_premask = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_logits = att_logits_premask.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att_logits, dim=-1)

        # Now calculate position of major attention based on softmax
        avg_logit = torch.sum(torch.mul(att_logits_premask, att), -1)  #  basically how confident each head is
        att_loc_avg = torch.sum(att * self.pos[0:T], -1)
        att_loc_rel = att_loc_avg - self.pos[0:T]
        eps = 1e-12
        att_loc_std = torch.pow(torch.sum(att * ((self.pos[0:T] - att_loc_avg.unsqueeze(-1)) ** 2 + eps), -1), 1.0/2.0)

        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Get attention location with correct concatenated values for each token.
        avg_logit = self.ln_importance(avg_logit.transpose(1, 2).contiguous().view(B, T, self.n_head))
        att_loc_rel = self.ln_rel(att_loc_rel.transpose(1, 2).contiguous().view(B, T, self.n_head))
        att_loc_std = self.ln_std(att_loc_std.transpose(1, 2).contiguous().view(B, T, self.n_head))
        # Get absolute location to send to the MLP as well
        # att_loc_orig = (torch.zeros((B,T,1), device=device) + (self.pos[0:T]).unsqueeze(-1)) * self.n_head # scale up
        att_loc = torch.cat((avg_logit, att_loc_rel, att_loc_std), dim=-1) # (B, T, n_head*3)

        head_scaling = 2.0 * self.mlp_scale_sigmoid(self.mlp_scale(att_loc)).transpose(-2, -1).unsqueeze(-1) # (B, nh, T, 1)
        y = y * head_scaling
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        self.print_num += 1
        if self.print_num == 400:
            # print('t0', att_loc[0,0,:])
            print('t-1 heads', head_scaling[0,:, -1,0])
            #print('t-1 locs', att_loc[0,-1,:])
            self.print_num=0

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
