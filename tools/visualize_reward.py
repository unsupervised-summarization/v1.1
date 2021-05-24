import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dist_samples = np.array([
    0.09902340157825938,
    -0.033671528663539285, 0.015581491236554888, 0.08652863690585208, 0.01969250378570898, -0.007285100616146144,
    -0.023572851375324017, -0.042553203096316695, -0.020289355691536236, 0.02168662873481022, -0.0347628087705102,
    -0.0698807162834947, 0.08775439336093539, 0.0201586059757844, -0.04114024739144507, -0.04297046710157668,
    -0.04023854412757619, -0.03557828337301629, 0.0299164840977486, 0.029427257255872388, -0.015835832089889288,
    0.06591911995440794, -0.034429259554474526, -0.010198718126230017, 0.015746401170716906, 0.017153329057451334,
    0.02259657317877435, -0.007710433576541174, 0.019627640607179586, 0.0768927167184097, -0.057162791468902366,
    -0.029962998827274222, 0.05383728676889838, -0.02747847436802021, -0.016919101801389416, -0.058939337436474074,
    0.02456124041455025, 0.03374838142462534, 0.031604429916772286, -0.03222378937027663, 0.019357529406028848,
    0.03833082317493317, 0.07790644259427282, -0.05696569391330785, -0.02871675621157959, 0.004505172975152494,
    -0.03976197697587643, 0.09864236289964368, -0.00043527698503620247, -0.019786442347753983, 0.03591160121412986,
    -0.0004398653261212409, -0.024597893796376676, 0.01396338272912824
])
minus = dist_samples.mean() - dist_samples.min()
plus = dist_samples.max() - dist_samples.mean()
print(minus, plus)

rewards = np.array([
    -0.4643218023460187,
    -0.4563871769579728,
    -0.2388018304383358,
    0.02920358300346564,
    -0.5706756227833188,
    -0.32929397673078975,
    -0.2180362826851721,
    -0.032063221115377835,
    0.14598607410099684,
    0.15702045972435533,
    0.2592768139878578,
    0.5140319421227925,
    -0.37855768701314496,
    0.06747250472056233,
    (0.2863868980921019+0.18726110825688613+0.31043760987175834)/3,
    -0.5195841267360338,
    -0.46017856676504765,
    -0.3315610362057528,
    -0.46017856676504765,
    -0.3315610362057528,
    (-0.35008712127083874+-0.15690010188014772)/2,
    -0.09078740643852083,
    0.0598707053453645,
    (0.26219526758379263+0.2787654145340172)/2,
    0.43451830599838104,
    0.6104143724289277,
    0.638185861708023,
    0.7538150571783839,
    0.8230036772262487,
    0.5921512097103907,
    (0.7471560164181268+0.7457117307868587)/2,
])
x = np.concatenate([np.arange(len(rewards))*4000, np.arange(len(rewards))*4000])
y = np.concatenate([rewards - (minus+plus), rewards + (minus+plus)])

sns.set_theme(style="whitegrid")

ax = sns.lineplot(x, y, color='mediumslateblue', label='reward')
ax.set(xlabel='n sentences', ylabel='reward (avg)')

plt.show()
