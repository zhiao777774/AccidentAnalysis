import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans


def SVM(df):
    df = df.copy()
    df = df.sample(10000)
    df.drop(['發生時間', '發生地點', '發生月份', '車種', '縣市'], inplace=True, axis=1)
    df['事故主要車種'].replace(['機車/腳踏車', '小型車', '大型車'], [0, 1, 2], inplace=True)
    df['城市規模'].replace(['一般縣市', '直轄市'], [0, 1], inplace=True)
    df['發生時段'].replace(['日間', '晚間', '夜間'], [0, 1, 2], inplace=True)
    df['死亡人數'].replace(np.nan, 0, inplace=True)
    x = df.drop(['死亡人數', '受傷人數'], axis=1)
    y = df[['死亡人數', '受傷人數']]

    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    svm = SVC(kernel='linear', random_state=0, gamma='scale', C=1, probability=True)
    svm.fit(x_train, y_train['受傷人數'])

    print('Accuracy(受傷人數SVM)', svm.score(x_test, y_test['受傷人數']))

    return svm
    

def KNN(df):
    df = df.copy()
    # df = df.sample(8000)
    df.drop(['發生時間', '發生地點', '發生月份', '車種', '縣市'], inplace=True, axis=1)
    df['事故主要車種'].replace(['機車/腳踏車', '小型車', '大型車'], [0, 1, 2], inplace=True)
    df['城市規模'].replace(['一般縣市', '直轄市'], [0, 1], inplace=True)
    df['發生時段'].replace(['日間', '晚間', '夜間'], [0, 1, 2], inplace=True)
    # df['死亡人數'].replace(np.nan, 0, inplace=True)
    x = df.drop(['死亡人數', '受傷人數'], axis=1)
    y = df[['死亡人數', '受傷人數']]

    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    dead_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p = 1).fit(x_train, y_train['死亡人數'])
    injured_knn = KNeighborsClassifier(n_neighbors=22, n_jobs=-1, p = 1).fit(x_train, y_train['受傷人數'])

    print('Accuracy(死亡人數)', dead_knn.score(x_test, y_test['死亡人數']))
    print('Accuracy(受傷人數)', injured_knn.score(x_test, y_test['受傷人數']))

    return dead_knn, injured_knn


def kmeans(df):
    df = df.copy()
    df.drop(['發生時間', '發生地點', '發生月份', '車種', '縣市'], inplace=True, axis=1)
    df['事故主要車種'].replace(['機車/腳踏車', '小型車', '大型車'], [0, 1, 2], inplace=True)
    df['城市規模'].replace(['一般縣市', '直轄市'], [0, 1], inplace=True)
    df['發生時段'].replace(['日間', '晚間', '夜間'], [0, 1, 2], inplace=True)
    x = df[['事故主要車種', '死亡人數', '城市規模']]

    km = KMeans(n_clusters=3,init='random',random_state=5)
    km.fit(x)

    plt.scatter(x['死亡人數'], x['事故主要車種'], c=km.predict(x))
    plt.savefig('./static/data/kmeans.png')

    return km