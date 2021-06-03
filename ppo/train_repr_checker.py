import torch
import torch.nn.functional as F
import joblib
import numpy as np

from repr_checker.model import tokenizer, model, optimizer
from repr_checker.data import DataLoader
from repr_checker.utils import Logger

device = torch.device('cuda')
model = model.to(device)

# from repr_checker.data import make_test
test_input_ids, test_attention_mask, test_labels = joblib.load('repr_checker/test.joblib')


EPOCH = 100
BATCHSIZE = 2

data_loader = DataLoader(tokenizer)
model.resize_token_embeddings(len(data_loader.tokenizer))

logger = Logger('train-loss', 'train-acc', 'test-loss', 'test-acc')

try:
    logger.load('repr_checker/train_logger.pkl')
    model.load_state_dict(torch.load('repr_checker/checkpoint.ckpt'))
    print('recover checkpoint')
    model.train()
except FileNotFoundError:
    print('failed to recover checkpoint')

for epoch in range(EPOCH):
    for iteration in range(1000):
        input_ids = []
        while len(input_ids) == 0:
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
        logger['train-loss'](loss)
        logger['train-acc'](acc)
        logger['test-loss'](None)
        logger['test-acc'](None)
        print('batch:', iteration, 'ㅤㅤ acc:', round(float(acc), 4), 'ㅤㅤ loss:', round(float(loss), 4), 'ㅤㅤ', end='\r')

        if iteration % 200 == 0:
            torch.save(model.state_dict(), 'repr_checker/checkpoint.ckpt')
            logger.save('repr_checker/train_logger.pkl')
            print('\n\nsaving model..\n')

        if iteration % 100 == 0:
            data_loader.load_file()
            with torch.no_grad():
                test_acc = []
                test_loss = []
                print()
                print('testing..')
                for i in range(0, len(test_input_ids), max(BATCHSIZE, 512)):
                    input_ids = torch.LongTensor(test_input_ids[i:i+BATCHSIZE]).to(device)
                    attention_mask = torch.LongTensor(test_attention_mask[i:i+BATCHSIZE]).to(device)
                    labels = torch.FloatTensor(test_labels[i:i+BATCHSIZE]).to(device)
                    print(input_ids.shape, attention_mask.shape)
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

        del input_ids, attention_mask, labels

    print(f'epoch{epoch}')
