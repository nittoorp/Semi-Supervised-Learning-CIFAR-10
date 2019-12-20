import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from fsl_project.cnn_util import Util
from fsl_project.generator import Generator

from fsl_project.cutmix import CutMixGenerator
from fsl_project.mixup import MixupGenerator

# credit : https://github.com/facebookresearch/mixup-cifar10

class MixAndCutGenerator(Generator):

    def generate_data(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        batch_index = batch_ids[:self.batch_size]
        rand_index = np.random.permutation(batch_index)
        X1 = self.X_train[batch_index]
        X2 = self.X_train[rand_index]
        y1 = self.y_train[batch_index]
        y2 = self.y_train[rand_index]
        lam = np.random.beta(self.alpha, self.alpha)

        bx1, by1, bx2, by2 = self.bbox_random(w, h, lam)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        y = y1 * lam + y2 * (1 - lam)

        if self.data_gen:
            i = 0
            while i < self.batch_size:
                X[i] = self.data_gen.random_transform(X[i])
                i = i + 1

        return X, y


def run_cutmix_algorithm(cnn_util):
    X_train, y_train, testX, testY = cnn_util.load_input_dataset()
    X_train, testX = cnn_util.prepare_img_pixels(X_train, testX)
    model = cnn_util.create_model()
    data_gen = ImageDataGenerator(rescale=1./255,)
    training_generator = CutMixGenerator(X_train, y_train, batch_size=64, alpha=0.2, data_gen=None).call()

    cutmix_obj = training_generator.gi_frame.f_locals.get('self')
    X_train_new = cutmix_obj.X_train
    y_train_new = cutmix_obj.y_train

    training_generator_new = MixupGenerator(X_train_new, y_train_new, batch_size=64, alpha=0.2, data_gen=None).call()

    history = model.fit_generator(generator=training_generator_new, steps_per_epoch=X_train.shape[0] // 64,validation_data=(testX, testY), epochs=1, verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)
    pickle.dump(model, open("Genericmodel.pkl", 'wb'))
    print('> %.3f' % (acc * 100.0))
    cnn_util.plot_results(history)


if __name__ == "__main__":
    cnn_util = Util()
    run_cutmix_algorithm(cnn_util)
