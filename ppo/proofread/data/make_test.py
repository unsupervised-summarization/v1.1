# Making test dataset
import joblib

from .dataloader import DataLoader
from ..model import tokenizer

data_loader = DataLoader(tokenizer)

with open('proofread/assets/dataset/wikicorpus/306.txt.test', 'r', encoding='utf8') as f:
    data = f.read()

data = data.split('\n')[:1000]
data_loader.data = data
test = data_loader.get_batch(1000)

print('saving')
joblib.dump(test, 'proofread/assets/dataset/test.joblib')
