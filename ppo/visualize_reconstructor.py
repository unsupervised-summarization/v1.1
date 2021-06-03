from reconstructor.utils import Logger
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1)
logger = Logger()
logger.load('reconstructor/pretrain_loss_log.pkl')
logger.plot(ax=axes[0], color='coral'   )

logger = Logger()
logger.load('reconstructor/pretrain_lr_log.pkl')
logger.plot(ax=axes[1])

plt.show()
