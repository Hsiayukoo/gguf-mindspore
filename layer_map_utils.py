"""
用于 layer 名称的映射
"""

ms_to_gguf_map = {
    # common
    "model.layers": "blk",
    "attention.wq": "attn_q",
    "attention.wk": "attn_k",
    "attention.wv": "attn_v",
    "attention.wo": "attn_output",
    "attention_norm": "attn_norm",
    "feed_forward.w1": "ffn.gate",
    "feed_forward.w3": "ffn_up",
    "feed_forward.w2": "ffn.down",
    "ffn_norm": "ffn_norm"
    # special
}