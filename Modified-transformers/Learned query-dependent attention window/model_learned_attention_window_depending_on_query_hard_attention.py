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
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.c_mem = nn.Linear(config.n_embd, config.n_head, bias=config.bias)
        self.mem_ReLU = nn.ReLU()
        self.register_buffer("rel_pos", -torch.flip(torch.cumsum(torch.flip(torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size), dims=[-1]), dim=-1), dims=[-1]) )

        self.config = config
        self.print_num=0

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        eps_prob = 1e-10

        ########### First get pseudo attention matrix of size T x T
        # calculate q_mem, k_mem, and q_size for all heads in batch
        q_size  = self.c_mem(x)
        q_size = q_size.view(B, T, self.n_head, 1).transpose(1, 2) # (B, nh, T, 1)
        q_size = 1.0 + (self.config.block_size / 2.0) * torch.exp(q_size)

        att_window_scaling = 1.0 - torch.exp( - self.mem_ReLU((self.rel_pos[:,:,:T,:T] + q_size)/q_size)) + eps_prob
        att_window_scaling = att_window_scaling.masked_fill(self.bias[:,:,:T,:T] == 0, 0.0)  # causal mask with 0s

        self.print_num+=1
        if self.print_num ==100:
            self.print_num=0
            whichtok=50
            numshow=15
            print('query size (attention window)', q_size[0,:, whichtok].view(-1))
            # print(att_window_scaling[0,0,whichtok,:numshow])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)

            # Add attention window mask
            att_lowerbounded = (att + eps_prob).masked_fill(self.bias[:,:,:T,:T] == 0, 0.0)
            att = (att_lowerbounded * att_window_scaling) / torch.sum(att_lowerbounded * att_window_scaling, dim=-1, keepdim=True)

            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
