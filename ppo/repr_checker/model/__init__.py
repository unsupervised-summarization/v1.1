from transformers import BartTokenizer, BartForSequenceClassification
from transformers import AdamW


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=1)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-7)
