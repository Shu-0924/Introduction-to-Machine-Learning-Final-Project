from keras.models import load_model
import numpy as np
import csv

MODEL_PATH = './model.h5'
TEST_DATA_PATH = './test.csv'
SUBMISSION_PATH = './submission.csv'


def load_data(path):
    x_data = []
    with open(path, encoding='utf-8-sig') as fin:
        csv_reader = csv.reader(fin)
        next(csv_reader)
        for row in csv_reader:
            info = [row[2]] + row[7:]
            x_data.append(info)
    return x_data


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


if __name__ == '__main__':
    m = load_model(MODEL_PATH)
    x_test = load_data(TEST_DATA_PATH)
    x_test = np.array(fill_blank(x_test))
    y_pred = m.predict(x_test, batch_size=128)
    with open(SUBMISSION_PATH, 'w', newline='') as fout:
        csv_writer = csv.writer(fout)
        csv_writer.writerow(['id', 'failure'])
        for idx, prob in enumerate(y_pred):
            csv_writer.writerow([idx+26570, prob[0]])
