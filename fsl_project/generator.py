from abc import abstractmethod
import numpy as np

class Generator:

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2,
                 shuffle=True, data_gen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.data_gen = data_gen

    def call(self):
        while True:
            indxs = self.define_order()
            a = len(indxs)
            b = self.batch_size * 2
            itr_num = int(a // b)
            i = 0
            while i < itr_num:
                start = i * self.batch_size * 2
                end = (i + 1) * self.batch_size * 2
                batch_ids = indxs[start: end]
                x, y = self.generate_data(batch_ids)
                i = i + 1
                yield x, y

    def define_order(self):
        i = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(i)
        return i

    def bbox_random(self, w, h, l):
        rx = np.random.randint(w)
        ry = np.random.randint(h)
        rl = np.sqrt(1 - l)
        rw = np.int(w * rl)
        rh = np.int(h * rl)
        bbx1 = np.clip(rx - rw // 2, 0, w)
        bby1 = np.clip(ry - rh // 2, 0, h)
        bbx2 = np.clip(rx + rw // 2, 0, w)
        bby2 = np.clip(ry + rh // 2, 0, h)
        return bbx1, bby1, bbx2, bby2

    @abstractmethod
    def generate_data(self, batch_ids):
        raise NotImplementedError("Must override generate_data")
