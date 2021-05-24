# Making test dataset
import joblib

from .dataloader import DataLoader, get_web_text
from ..model import tokenizer

data_loader = DataLoader(tokenizer)

data_loader.n_range = (3072, 3074)
test = data_loader.get_batch(1000)

print(test[0].shape)

print('saving')
joblib.dump(test, 'repr_checker/test.joblib')
