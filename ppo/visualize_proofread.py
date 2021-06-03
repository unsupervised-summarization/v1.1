from proofread.utils.logger import Logger
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1)
logger = Logger()
logger.load('proofread/train_logger.pkl')

logger['train-loss'].plot(ax=axes[0], color='coral')
logger['train-acc'].plot(ax=axes[1])
logger['test-loss'].plot(ax=axes[0], color='darkorchid')
logger['test-acc'].plot(ax=axes[1], color='limegreen')

plt.show()
