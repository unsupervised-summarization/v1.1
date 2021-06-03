import torch
import torch.nn.functional as F
import joblib
import numpy as np

from proofread.model import tokenizer, model, optimizer
from proofread.data import DataLoader
from proofread.utils import Logger

device = torch.device('cuda')
model = model.to(device)

try:
    model.load_state_dict(torch.load('proofread/checkpoint.ckpt'))
except FileNotFoundError:
    print("failed to load checkpoint in train_proofread.py")
model.train()

test_input_ids, test_attention_mask, test_labels = joblib.load('proofread/assets/dataset/test.joblib')

EPOCH = 100
BATCHSIZE = 30

data_loader = DataLoader(tokenizer)
logger = Logger('train-loss', 'train-acc', 'test-loss', 'test-acc')

for epoch in range(EPOCH):
    for iteration in range(1000):
        input_ids, attention_mask, labels = data_loader.get_batch(BATCHSIZE)

        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
        labels = torch.FloatTensor(labels).to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        out = torch.sigmoid(outputs[0])

        loss = F.binary_cross_entropy(out, labels)

        loss.backward()

        optimizer.step()

        acc = (labels.cpu().detach().numpy().astype(int) == (out.cpu().detach().numpy() >= 0.5).astype(int)).sum() / len(labels)
        loss = loss.cpu().detach().numpy()
        print('batch:', iteration, 'ㅤㅤacc:', round(float(acc), 4), 'ㅤㅤloss:', round(float(loss), 4), 'ㅤㅤ', end='\r')

        logger['train-loss'](loss)
        logger['train-acc'](acc)
        logger['test-loss'](None)
        logger['test-acc'](None)

        if iteration % 200 == 0:
            torch.save(model.state_dict(), 'proofread/checkpoint.ckpt')
            logger.save('proofread/train_logger.pkl')
            print('\n\nsaving model..\n')

        if iteration % 100 == 0:
            data_loader.load_file()
            model.eval()
            with torch.no_grad():
                test_acc = []
                test_loss = []
                print()
                print('testing..')
                for i in range(0, len(test_input_ids), BATCHSIZE):
                    input_ids = torch.LongTensor(test_input_ids[i:i+BATCHSIZE]).to(device)
                    attention_mask = torch.LongTensor(test_attention_mask[i:i+BATCHSIZE]).to(device)
                    labels = torch.FloatTensor(test_labels[i:i+BATCHSIZE]).to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    out = torch.sigmoid(outputs[0])
                    loss = F.binary_cross_entropy(out, labels)
                    acc = (labels.cpu().detach().numpy().astype(int) == (out.cpu().detach().numpy() >= 0.5).astype(int)).sum() / len(labels)
                    loss = loss.cpu().detach().numpy()
                    test_acc.append(acc)
                    test_loss.append(loss)
                    print(i, end='\r')
                print('test acc', np.mean(test_acc))
                print('test loss', np.mean(test_loss))
                logger['test-loss'](test_loss)
                logger['test-acc'](test_acc)
                print()
            model.train()

        del input_ids, attention_mask, labels

    print(f'epoch{epoch}')
