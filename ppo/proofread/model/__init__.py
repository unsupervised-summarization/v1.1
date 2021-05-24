from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AdamW


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-7)
