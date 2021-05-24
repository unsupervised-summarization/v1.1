# Actor model
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification


actor_model_name: str = "gpt2"
critic_model_name: str = "distilgpt2"


def load_actor_model():
    actor_model = GPT2LMHeadModel.from_pretrained(actor_model_name)  # model (LM)
    actor_model.train()
    return actor_model


def load_critic_model():
    critic_model = GPT2ForSequenceClassification.from_pretrained(critic_model_name, num_labels=1)  # model (regression)
    critic_model.train()
    return critic_model


def load_tokenizer():
    # gpt2 tokenizer == distilgpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(actor_model_name)  # tokenizer
    return tokenizer
