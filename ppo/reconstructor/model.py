from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .args import args


model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args['num_warmup_steps'],
    num_training_steps=args['epochs']*1000-args['num_warmup_steps'],
    num_cycles=5,
)
