import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from fsl_project.cnn_util import Util
from fsl_project.generator import Generator


# Credit  : https://github.com/facebookresearch/mixup-cifar10
class MixupGenerator(Generator):

    def generate_data(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.data_gen:
            i = 0
            while i < self.batch_size:
                X[i] = self.data_gen.random_transform(X[i])
                X[i] = self.data_gen.standardize(X[i])
                i = i+1

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

    
def run_mixup_algorithm(cnn_util):
    X_train, y_train, testX, testY = cnn_util.load_input_dataset()
    X_train, testX = cnn_util.prepare_img_pixels(X_train, testX)
    model = cnn_util.create_model()
    data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    training_generator = MixupGenerator(X_train, y_train, batch_size=64, alpha=0.2, data_gen=None).call()
    history = model.fit_generator(generator=training_generator, steps_per_epoch=X_train.shape[0] // 64, validation_data=(testX, testY), epochs=100, verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)
    pickle.dump(model, open("Genericmodel.pkl", 'wb'))
    print('> %.3f' % (acc * 100.0))
    cnn_util.plot_results(history)


if __name__ == "__main__":
    cnn_util = Util()
    run_mixup_algorithm(cnn_util)
