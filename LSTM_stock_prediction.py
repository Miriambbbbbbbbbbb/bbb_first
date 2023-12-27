import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
from math import sqrt
import math
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D ,MaxPooling1D
from keras.layers import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer
from sklearn.metrics import mean_absolute_error

# 加载数据
df = pd.read_csv("D:\\研究生课件\\open_data_source\\final_project\\FinalProjectDemo\\FinalProjectDemo\\dataset\\train.csv")

# 数据为column名称为open, close, low, high的dataframe, date为索引
df.drop(df.columns[[5, 6]], axis=1, inplace=True)
df.set_index("Date", inplace=True)
df['High'] = df['High'] / 10000
df['Open'] = df['Open'] / 10000
df['Low'] = df['Low'] / 10000
df['Close'] = df['Close'] / 10000

# 转换为ndarray
data = df.to_numpy()

# 设定华东窗口，result为要输入LSTM的值
result = []
sequence_length = 6
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.array(result)

row = round(0.8 * result.shape[0])

# 设置训练集数据
train = result[:, :]

x_train = train[:, :-1]
y_train = train[:, -1][:, -1]

amount_of_features = len(df.columns)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

# 建立LSTM模型
def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(LSTM(32, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(d))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


model = build_model([4,5,1])
start = timer()
# 训练模型
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=25,
                    validation_split=0.2,
                    verbose=2)
end = timer()

history_dict = history.history

mae = history_dict['mae']
epochs = range(1, len(mae) + 1)


df2 = pd.read_csv("D:\\研究生课件\\open_data_source\\final_project\\FinalProjectDemo\\FinalProjectDemo\\dataset\\val.csv")

# 数据为column名称为open, close, low, high的dataframe, date为索引
df2.drop(df2.columns[[5, 6]], axis=1, inplace=True)
df2.set_index("Date", inplace=True)

# 转换为ndarray
data2 = df2.to_numpy()

# 设定华东窗口，result为要输入LSTM的值
result2 = []

sequence_length2 = 6
for index in range(len(data2) - sequence_length):
    result2.append(data2[index: index + sequence_length2])
result2 = np.array(result2)

x_test = result2[:, :-1]
y_test = result2[:, -1][:,-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

x_test = x_test/10000
p = model.predict(x_test)
y = y_test
y_pred = p.reshape(800)
y_pred = y_pred * 10000