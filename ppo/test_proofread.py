import torch
import numpy as np

from proofread.model import tokenizer, model, optimizer
from proofread.data import DataLoader

device = torch.device('cuda')
model = model.to(device)

data_loader = DataLoader(tokenizer)

model.load_state_dict(torch.load('proofread/checkpoint.ckpt'))
model.eval()


def predict(text: str) -> np.ndarray:
    text = data_loader.preprocessing(text)
    encoding = tokenizer([text], padding=True, truncation=True)
    input_ids = torch.LongTensor(encoding['input_ids']).to(device)
    attention_mask = torch.LongTensor(encoding['attention_mask']).to(device)

    out = model(input_ids, attention_mask=attention_mask)[0]
    out = torch.sigmoid(out)
    return out.cpu().detach().numpy()


def lime(text: str) -> np.ndarray:
    org_out = predict(text)
    print(org_out)
    encoding = tokenizer([text])
    tokens = [tk.replace('Ġ', '_') for tk in tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])]
    print(tokens)
    for i in range(len(encoding['input_ids'][0])):
        mask = encoding['attention_mask'].copy()
        mask[0][i] = 0
        input_ids = torch.LongTensor(encoding['input_ids']).to(device)
        attention_mask = torch.LongTensor(mask).to(device)

        out = model(input_ids, attention_mask=attention_mask)[0]
        out = torch.sigmoid(out).cpu().detach().numpy()
        diff = (org_out - out) ** 2
        print(diff[0][0], tokens[i], out)


sentence = 'Turbine experts suggested it was a simple mechanical failure. The plot thickened further Friday, with The Sun saying it had been"bombarded" with reports of UFO sightings from hundreds of witnesses in the area where was the turbine destroyed.'

lime(sentence)


print()
print('0: uncorrect sentence')
print('1: perfect sentence')
print()
while True:
    text = input('Enter Sentence :')
    print(predict(text).reshape(-1))
