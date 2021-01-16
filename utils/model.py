import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


def train(model_path, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = tf.keras.Sequential()
    model.add(layers.Dense(27, activation='relu', input_dim=3))
    model.add(layers.Dense(27, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=200, batch_size=1000)

    model.summary()
    model.save(model_path)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    return model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


if __name__ == '__main__':
    from preprocess import TMA1, TMA2, TMA3

    df = pd.concat([TMA1, TMA2], ignore_index=True).copy()
    df = df.sample(100000)

    df.drop(['發生時間', '發生地點', '發生月份', '車種', '縣市'], inplace=True, axis=1)
    df['事故主要車種'].replace(['機車/腳踏車', '小型車', '大型車'], [0, 1, 2], inplace=True)
    df['城市規模'].replace(['一般縣市', '直轄市'], [0, 1], inplace=True)
    df['發生時段'].replace(['日間', '晚間', '夜間'], [0, 1, 2], inplace=True)
    df['死亡人數'].replace(np.nan, 0, inplace=True)

    x = df.drop(['死亡人數', '受傷人數'], axis=1)
    y = df[['死亡人數', '受傷人數']]
    y = pd.get_dummies(y, columns=['受傷人數'], prefix='injured_')
    y = y.drop('死亡人數', axis=1)

    train('./static/data/model4', x, y)