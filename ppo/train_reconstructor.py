# Pretrain reconstructor
from reconstructor.trainer import Trainer
from reconstructor.data import DataLoader
from reconstructor.args import args


trainer = Trainer()
data_loader = DataLoader(trainer.tokenizer)

for epoch in range(args['epochs']):
    for iteration in range(1000):
        summaries_tokens, documents_tokens = data_loader.get_batch(args['batch_size'])

        trainer.train_step(summaries_tokens, documents_tokens)  # train
        print('loss:', trainer.loss_logger[-1], 'iter:', iteration, 'lr:', trainer.lr_logger[-1], '....', end='\r')

        if iteration % 500 == 0:
            # save model
            print()
            print('saving model...')
            print()
            trainer.save('reconstructor/pretrained.ckpt')
            trainer.loss_logger.save('reconstructor/pretrain_loss_log.pkl')
            trainer.lr_logger.save('reconstructor/pretrain_lr_log.pkl')

"""
summaries = ['This is my name', 'Perfection is really motivating.']
documents = ['Hello my name is Jun Because of my new camera', 'Also this is a perfect example so I am really happy']

# preprocessing
summaries = [sent.lower() for sent in summaries]
documents = [sent.lower() for sent in documents]

# tokenizing
summaries_tokens = [trainer.tokenizer.encode(sent) for sent in summaries]
documents_tokens = [trainer.tokenizer.encode(sent) for sent in documents]

# one step for training
try:
    for i in range(10000):
        trainer.train_step(summaries_tokens, documents_tokens)
except KeyboardInterrupt:
    pass

print(trainer.loss_logger)

#import torch
#trainer.model.to('cpu')
#print(torch.argmax(trainer.model(**trainer.tokenizer('This is my<||>', return_tensors='pt'))[0], dim=-1))
trainer.eval(summaries_tokens, documents_tokens)
trainer.eval(summaries_tokens, [i[:-1] for i in documents_tokens])

trainer.loss_logger.plot()
"""
