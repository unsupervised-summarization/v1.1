args = {
    'device': 'cuda',
    'max_length': 1024,  # Max
    'drop_long_texts': True,  # Drop texts longer than {max_length} tokens
    'epochs': 100,
    'batch_size': 1,
    'num_warmup_steps': 100,
}
