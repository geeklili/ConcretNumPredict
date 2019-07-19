from collections import defaultdict
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from tail_line import tail_line


def ridge_regression():
    data = pd.read_csv('./data/train_data.csv')
    # data = data.iloc[:10000, :]

    # print(data)
    train_data_x = data.iloc[:, 1:-1]
    train_data_y = data.iloc[:, 162:]
    train_data_x = train_data_x.values

    train_data_y['162'] = (train_data_y - np.min(train_data_y)) / (np.max(train_data_y) - np.min(train_data_y))
    train_data_y2 = np.array(train_data_y['162'].tolist())

    x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y2, test_size=0.2)
    # 需要做标准化处理对于特征值处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 使用岭回归进行预测
    rd = Ridge(alpha=1.0)
    # rd = LogisticRegression()

    rd.fit(x_train, y_train)
    # print("岭回归的权重参数为：", rd.coef_)

    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    y_rd_predict = rd.predict(x_test)
    tail_line()
    print("岭回归的预测的结果为：\n", y_rd_predict)

    # 怎么评判这两个哪个好？
    print("岭回归的均方误差为：\n", mean_squared_error(y_test, y_rd_predict))

    y_rd_predict = y_rd_predict * (np.max(train_data_y['161']) - np.min(train_data_y['161'])) + np.min(
        train_data_y['161'])
    y_test = y_test * (np.max(train_data_y['161']) - np.min(train_data_y['161'])) + np.min(train_data_y['161'])
    print(y_rd_predict)
    print(y_test)

    percent = [(y_rd_predict[a] - b) / b for a, b in enumerate(y_test)]
    di = defaultdict(int)
    for i in percent:
        # if -200 < i < 200:
        di[round(i, 2)] += 1
    dis = {a: b for a, b in sorted(di.items(), key=lambda x: x[1], reverse=True)}
    dis2 = {a: b for a, b in sorted(di.items(), key=lambda x: x[1], reverse=True) if a <= 0.3}
    per = sum(dis2.values()) / sum(dis.values())
    print(di)
    tail_line()
    print("概率为：", per)
    t = sorted(di.items(), key=lambda x: x[0], reverse=True)
    plt.plot([a for a, b in t], [b for a, b in t])
    plt.show()
    tail_line()


def neural_network_regression():
    n_dim = 368
    data = pd.read_csv('./data/train_data_salary.csv')
    # data = data.iloc[:10000, :]

    # print(data)
    train_data_x = data.iloc[:, 1:-1]
    train_data_y = data.iloc[:, n_dim:]
    train_data_x = train_data_x.values

    train_data_y[str(n_dim)] = (train_data_y - np.min(train_data_y)) / (np.max(train_data_y) - np.min(train_data_y))
    train_data_y2 = np.array(train_data_y[str(n_dim)].tolist())

    max_val = np.max(train_data_y[str(n_dim - 1)])
    min_val = np.min(train_data_y[str(n_dim - 1)])
    print(max_val)
    print(min_val)

    x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y2, test_size=0.2)
    # 需要做标准化处理对于特征值处理
    # try:
    #     std_x = joblib.load('./data/model/std2.model')
    # except Exception as e:
    #     std_x = StandardScaler()
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    joblib.dump(std_x, './data/model/std2.model')

    # 创建模型，Sequential(): 多个网络层的线性堆叠模型
    model = Sequential()

    # 添加神经网络层，并指定，input_dim：输入维度的个数，units：神经元的个数

    model.add(Dense(input_dim=n_dim-1, units=256, activation='relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(input_dim=n_dim - 1, units=256))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, ))
    rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    # 选择损失函数，优化器
    model.compile(optimizer=rmsprop, loss='mse'),  # optimizer='sgd')

    model.fit(x_train, y_train, epochs=50, batch_size=100)
    print('testing...')
    # 评估测试集
    cost = model.evaluate(x_test, y_test, batch_size=400)
    tail_line()
    print('test cost is: ', cost)
    y_predict = model.predict(x_test)
    y_predict = np.array([i[0] for i in y_predict])
    y_rd_predict = y_predict * (np.max(train_data_y[str(n_dim - 1)]) - np.min(train_data_y[str(n_dim - 1)])) + np.min(
        train_data_y[str(n_dim - 1)])
    y_test = y_test * (np.max(train_data_y[str(n_dim - 1)]) - np.min(train_data_y[str(n_dim - 1)])) + np.min(
        train_data_y[str(n_dim - 1)])
    print(y_rd_predict)
    print(y_test)

    percent = [(y_rd_predict[a] - b) / y_rd_predict[a] for a, b in enumerate(y_test)]
    di = defaultdict(int)
    for i in percent:
        # if -200 < i < 200:
        di[round(i, 2)] += 1
    dis = {a: b for a, b in sorted(di.items(), key=lambda x: x[1], reverse=True)}
    dis2 = {a: b for a, b in sorted(di.items(), key=lambda x: x[1], reverse=True) if a <= 0.3}
    per = sum(dis2.values()) / sum(dis.values())
    print(di)
    tail_line()
    print("概率为：", per)
    t = sorted(di.items(), key=lambda x: x[0], reverse=True)
    t = [(a, b) for a, b in t if -5 <= a <= 5]
    plt.plot([a for a, b in t], [b for a, b in t])
    plt.show()
    tail_line()
    model.save('./data/model/neural_network_regression.model')
    # model = keras.models.load_model('./data/model/neural_network_regression.model')


def neural_network():
    n_dim = 156
    data = pd.read_csv('./data/train_data_category.csv')
    # data = data.iloc[:1000, :]

    # print(data)
    # train_data_x = data.iloc[:, 1:-1]
    train_data_x = data.iloc[:, 1:n_dim]
    train_data_y = data.iloc[:, n_dim:]
    train_data_x = train_data_x.values
    train_data_y = pd.get_dummies(train_data_y[str(n_dim - 1)]).values
    # train_data_y = np.array(train_data_y['315'].tolist())

    X_train, x_test, Y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=0.2)
    # x_test = train_data_x
    print(x_test)
    # 需要做标准化处理对于特征值处理
    # try:
    #     std_x = joblib.load('./data/model/std.model')
    # except Exception as e:
    #     std_x = StandardScaler()
    std_x = StandardScaler()
    X_train = std_x.fit_transform(X_train)

    x_test = std_x.transform(x_test)

    joblib.dump(std_x, './data/model/std.model')
    # print(x_train)

    # 构建神经网络
    model = Sequential([
        Dense(units=512, input_dim=n_dim - 1),
        Activation('relu'),
        Dropout(0.2),
        Dense(units=256),
        Activation('relu'),
        Dropout(0.2),
        Dense(units=6),
        Activation('softmax'),
    ])

    # 定义自己的优化器:optimizer
    rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)

    # 定义模型的损失函数，优化器
    model.compile(
        optimizer=rmsprop,  # 此处也可以将：optimizer='rmsprop', 如果是字符串的rmsprop就是默认的没有改动的优化器
        loss='categorical_crossentropy',  # 交叉熵损失
        metrics=['accuracy']
    )

    # 训练数据集: epochs: 训练几个轮回， batch_size: 一次训练多少数据
    model.fit(X_train, Y_train, epochs=50, batch_size=32)

    # 测试数据
    model.save('./data/model/keras_salary_predict.model')
    model = keras.models.load_model('./data/model/keras_salary_predict.model')
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy:', accuracy)
    # print(x_test)
    # ret = model.predict(x_test)
    # model.
    # print(ret)
    # for j in ret:
    #     print(list([round(i*100, 2) for i in j]))


if __name__ == "__main__":
    pass
    neural_network_regression()
    # ridge_regression()
    # neural_network()
