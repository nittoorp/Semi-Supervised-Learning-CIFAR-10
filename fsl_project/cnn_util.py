import sys
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import pickle
from keras.preprocessing.image import ImageDataGenerator


class Util:

    def load_input_dataset(self):
        (a, b), (c, d) = cifar10.load_data()
        b = to_categorical(b)
        d = to_categorical(d)
        print("loading done")
        return a, b, c, d

    def prepare_img_pixels(self,train, test):
        a = train.astype('float32')
        b = test.astype('float32')
        a = a / 255.0
        b = b / 255.0
        print("normalize done")
        return a, b

    def create_model(self):
        mdl = Sequential()
        mdl.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        mdl.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        mdl.add(MaxPooling2D((2, 2)))
        mdl.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        mdl.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        mdl.add(MaxPooling2D((2, 2)))
        mdl.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        mdl.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        mdl.add(MaxPooling2D((2, 2)))
        mdl.add(Flatten())
        mdl.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        mdl.add(Dense(10, activation='softmax'))
        opt = SGD(lr=0.001, momentum=0.9)
        mdl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        print("model done")
        print(mdl.summary())
        return mdl

    def plot_results(self, hstry):
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(hstry.history['loss'], color='blue', label='train')
        plt.plot(hstry.history['val_loss'], color='orange', label='test')
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(hstry.history['acc'], color='blue', label='train')
        plt.plot(hstry.history['val_acc'], color='orange', label='test')
        filename = sys.argv[0].split('/')[-1]
        plt.savefig(filename + '_plot.png')
        plt.close()


    def run_algorithm(self):
        X_train, y_train, testX, testY = self.load_input_dataset()
        X_train, testX = self.prepare_img_pixels(X_train, testX)
        model = self.create_model()
        training_generator = self.get_generator_object(X_train, y_train)
        history = model.fit_generator(generator=training_generator, steps_per_epoch=X_train.shape[0] // 64,
                                      validation_data=(testX, testY), epochs=1, verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        pickle.dump(model, open("Genericmodel.pkl", 'wb'))
        print('> %.3f' % (acc * 100.0))
        self.plot_results(history)

