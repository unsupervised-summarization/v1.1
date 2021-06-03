import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import logger.py
with open('ppo/reconstructor/utils/logger.py', 'r') as f:
    code = f.read()
exec(code)

logger = Logger()
logger.load('env_logger2.pkl')

names = ['repr_checker', 'proofread', 'length', 'reconstruct']

mode = 0

if mode == 0:
    fig = plt.figure()
    gs = GridSpec(ncols=2, nrows=3)  # width, height
    logger[names[0]].plot(ax=fig.add_subplot(gs[0, 0]))
    logger[names[1]].plot(ax=fig.add_subplot(gs[0, 1]))
    logger[names[2]].plot(ax=fig.add_subplot(gs[1, 0]))
    logger[names[3]].plot(ax=fig.add_subplot(gs[1, 1]))

    logger = Logger()
    logger.load('env_logger.pkl')
    logger.name = "final reward"
    logger.plot(ax=fig.add_subplot(gs[2, :]), color="#12b886")

    plt.show()

    print(logger)
else:
    fig = plt.figure()
    gs = GridSpec(ncols=2, nrows=1)  # width, height
    logger[names[0]].plot(ax=fig.add_subplot(gs[0, 0]))
    logger[names[1]].plot(ax=fig.add_subplot(gs[0, 0]), color="green")
    logger[names[2]].plot(ax=fig.add_subplot(gs[0, 0]), color="coral")
    logger[names[3]].plot(ax=fig.add_subplot(gs[0, 0]), color="blue")

    logger = Logger()
    logger.load('env_logger.pkl')
    logger.name = "final reward"
    logger.plot(ax=fig.add_subplot(gs[0, 1]), color="#12b886")

    plt.show()