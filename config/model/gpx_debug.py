# baby GPX dense model for debugging purposes

model_type = 'gpx-dense'

model_args = dict(
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=False,
    rms_eps=1e-6  # epsilon for RMSNorm
)