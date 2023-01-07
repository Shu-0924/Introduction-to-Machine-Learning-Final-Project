from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import AUC
import numpy as np
import csv

MODEL_PATH = './model.h5'
TRAIN_DATA_PATH = './train.csv'


def load_data(path):
    x_data, y_data = [], []
    with open(path, encoding='utf-8-sig') as fin:
        csv_reader = csv.reader(fin)
        next(csv_reader)
        for row in csv_reader:
            info = [row[2]] + row[7:]
            x_data.append(info[:-1])
            y_data.append(info[-1])
    return x_data, y_data


def fill_blank(x_data):
    column_cnt = [0] * len(x_data[0])
    column_mean = [0] * len(x_data[0])
    for i in range(len(x_data[0])):
        for row in x_data:
            if row[i] != '':
                column_mean[i] += float(row[i])
                column_cnt[i] += 1
    for i in range(len(x_data[0])):
        column_mean[i] /= column_cnt[i]

    new_data = []
    for row in x_data:
        new_data.append([])
        for i, num in enumerate(row):
            if num != '':
                new_data[-1].append(float(num))
            else:
                new_data[-1].append(column_mean[i])
    return new_data


def build_model(column_num):
    model = Sequential()
    model.add(Input(column_num))
    model.add(Dense(column_num))
    model.add(Dense(column_num))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Dense(4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[AUC()])
    model.summary()
    return model


if __name__ == '__main__':
    X, Y = load_data(TRAIN_DATA_PATH)
    X = np.array(fill_blank(X))
    Y = np.array([[int(failure)] for failure in Y])

    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    x_train, y_train = X[idxs], Y[idxs]

    m = build_model(x_train.shape[1])
    m.fit(x_train, y_train, batch_size=128, epochs=50)
    m.save(MODEL_PATH)
