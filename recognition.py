import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from sklearn.model_selection import train_test_split

DATA = 'data.npy'
TARGET = 'target.npy'


# 获取训练集与测试集
def get_train_test(split_ratio=.6, random_state=42):
    X = np.load(DATA)
    y = np.load(TARGET)
    assert X.shape[0] == y.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state,
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test


def main():
    x_train, x_test, y_train, y_test = get_train_test()
    
    x_train = x_train.reshape(-1, 11*20)
    x_test = x_test.reshape(-1, 11*20)
    
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    print(y_test_hot)
    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(220,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    print('-----------')
    
    result = model.fit(x_train, y_train_hot, batch_size=100, epochs=20, verbose=1,
                        validation_data=(x_test, y_test_hot))
    print(result)
    print(result.history)
    plot_history(result)


def plot_history(result):
    plt.plot(result.history['accuracy'],label='train')
    plt.plot(result.history['val_accuracy'],label='validation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
