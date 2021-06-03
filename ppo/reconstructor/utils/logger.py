from typing import List, Dict
import datetime
import pickle
import seaborn as sns
import numpy as np


class Logger:
    def __init__(self, *names):
        assert '<SETS>' not in names

        self.loggers: Dict[str, Logger] = {}
        self.logs: List[Dict] = []

        if len(names) == 1:
            self.name = names[0]
        elif len(names) == 0:
            self.name = 'unnamed-logger'
        else:
            self.name = "<SETS>"  # multiple loggers
            self.loggers: Dict = {name: Logger(name) for name in names}

    def __call__(self, data, date=None):
        if self.name == "<SETS>":
            raise Exception("You can't append logs into a set of multiple loggers.")

        if date is None:
            date = datetime.datetime.now()

        unit = {
            'when': date,
            'data': data
        }

        self.logs.append(unit)

    def log(self, *args, **kwargs):
        return self(*args, **kwargs)

    @staticmethod
    def strftime(date: datetime.datetime) -> str:
        return date.strftime("%m/%d %H:%M:%S")

    def unit_to_str(self, unit: Dict) -> str:
        date: str = self.strftime(unit['when'])
        return f"({date}) {unit['data']}"

    def __str__(self):
        if self.name == "<SETS>":
            return "<a set of loggers: " + ', '.join(self.loggers.keys()) + '>'

        prefix = self.name + '\n' if self.name is not None else ''
        if len(self.logs) >= 10:
            return prefix + \
                   '\n'.join(map(self.unit_to_str, self.logs[:5])) + \
                   '\n...\n' + \
                   '\n'.join(map(self.unit_to_str, self.logs[-5:]))
        else:
            return prefix + \
                   '\n'.join(map(self.unit_to_str, self.logs))

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if self.name == "<SETS>":
            return self.loggers[item]
        else:
            return self.logs[item]['data']

    def save(self, path: str = None):
        # save logs
        if path is None:
            path = f'{self.name}-log.pkl'

        with open(path, 'wb') as f:
            pickle.dump((self.name, self.loggers, self.logs), f)

    def load(self, path: str = None):
        # load logs
        if path is None:
            path = f'{self.name}-log.pkl'

        with open(path, 'rb') as f:
            d = pickle.load(f)

        try:
            self.name, self.loggers, self.logs = d
        except ValueError:
            self.name, self.logs = d

    def get_data(self) -> List:
        # Return list of logs.
        if self.name == "<SETS>":
            raise Exception("You can't get logs from a set of multiple loggers.")
        return [i['data'] for i in self.logs]

    @staticmethod
    def rolling_window(a: np.ndarray, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides).mean(axis=-1)

    def all_to_float(self) -> (List[int], List[float]):
        # convert all data of logs to float type.
        index = []
        data = []
        n = 0
        for i, j in enumerate(self.logs):
            if j['data'] is None:
                n += 1
            else:
                index.append(n)
                n += 1
                try:
                    data.append(float(j['data']))
                except TypeError:
                    data.append(float(np.mean(j['data'])))
        return index, data

    def plot(self, ax=None, rolling_window=None, plot_type='line', theme="whitegrid", color="mediumslateblue", label=None):
        # Plot logs.
        # To use this function, Logs should be series of number.
        if self.name == "<SETS>":
            raise Exception("You can't plot a set of multiple loggers.")

        if isinstance(self.logs[0]['data'], list):
            # draw confidence band
            index, data = [], []
            for i in range(len(self.logs)):
                tmp = list(self.logs[i]['data'])
                data.extend(tmp)
                index.extend([i for _ in range(len(tmp))])
            if len(data) >= 500 and rolling_window is not False:
                # rolling window
                data2 = data.copy()
                index = []
                rolling_window = 20
                data = []
                tmp = []
                i = 0
                for d in data2:
                    tmp.append(d)
                    if len(tmp) >= len(self.logs[0]['data'])*rolling_window:
                        data.extend(tmp)
                        index.extend([i for _ in range(len(tmp))])
                        i += 1
                        tmp = []
            rolling_window = False
        else:
            index, data = self.all_to_float()
            print(len(index), len(data))
            if label is None:
                label = self.name
            if rolling_window is None:
                # rolling_window == False -> don't show moving average
                # rolling_window == None -> automatic rolling_window size
                # rolling_window == {int} -> rolling_window size
                if len(data) >= 100:
                    rolling_window = int(np.sqrt(len(data)))
                else:
                    rolling_window = False

        if theme is not None:
            sns.set_theme(style=theme)
            if ax is not None:
                ax.set_title(self.name)
                ax.grid(True)

        if plot_type == 'line':
            if not rolling_window:
                axes = sns.lineplot(x=index, y=data, color=color, label=label, ax=ax)
            else:
                axes = sns.lineplot(x=index, y=data, color=color, alpha=0.3, ax=ax)
                roll = self.rolling_window(np.array(data), rolling_window)
                print(len(roll), len(index))
                sns.lineplot(x=index[int(np.floor((len(index)-len(roll))/2)):-int(np.ceil((len(index)-len(roll))/2))], y=roll, color=color, label=label, ax=ax)
        elif plot_type == 'scatter':
            axes = sns.scatterplot(x=index, y=data, color=color, label=label, ax=ax)
        else:
            raise Exception(f"Unknown Plot Type: {plot_type}. please select plot type ")

        axes.set(ylabel=label)
        return axes
