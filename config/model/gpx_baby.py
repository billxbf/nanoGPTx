# baby GPX dense model for debugging purposes

model_type = 'gpx-dense'

model_args = dict(
    n_layer=6,
    n_head=6,
    n_kv_head=2,  # number of key-value heads, must divide n_head
    n_embd=384,
    dropout=0.2,
    bias=False,  # bias for linear layers
    rms_eps=1e-6,  # epsilon for RMSNorm
    rope_theta=10000,
    max_position_embeddings=2048, 
)