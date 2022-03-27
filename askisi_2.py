import numpy as np
import pandas as pd

from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

df_timeseries = pd.read_excel('/Users/john/Downloads/data_akbilgic.xlsx', header=1)
df_timeseries = df_timeseries.iloc[0:529, :]
df_timeseries.drop(['date', 'ISE'], axis=1, inplace=True)
scaler = StandardScaler()
df_standard = pd.DataFrame(scaler.fit_transform(df_timeseries.iloc[:, 1:].values),
                           columns=df_timeseries.columns[1:])

df_standard['price'] = df_timeseries['ISE.1']


def timeseries_to_supervised(df, n_in, n_out):
    agg = pd.DataFrame()

    for i in range(n_in, 0, -1):
        df_shifted = df.shift(i).copy()
        df_shifted.rename(columns=lambda x: ('%s(t-%d)' % (x, i)), inplace=True)
        agg = pd.concat([agg, df_shifted], axis=1)

    for i in range(0, n_out):
        df_shifted = df.shift(-i).copy()
        if i == 0:
            df_shifted.rename(columns=lambda x: ('%s(t)' % (x)), inplace=True)
        else:
            df_shifted.rename(columns=lambda x: ('%s(t+%d)' % (x, i)), inplace=True)
        agg = pd.concat([agg, df_shifted], axis=1)
    agg.dropna(inplace=True)
    return agg

for ii in [1,3,6]:
    n_in = ii
    n_out = 1
    sdf = timeseries_to_supervised(df_standard, n_in, n_out)
    print(sdf.columns)

    data = shuffle(sdf.values, random_state=0)
    X = data[:, :-1]
    y = data[:, -1]

    len_data = sdf.shape[0]
    print(len_data)
    train_size = int(len_data * .8)
    test_size = int(len_data * .2)
    print("Train size: %d" % train_size)
    print("Test size: %d" % test_size)

    xtr = X[:train_size, :]
    ytr = y[:train_size]

    xte = X[train_size:, :]
    yte = y[train_size:]

    model = Sequential()

    model.add(Dense(units=len(xtr), input_dim=sdf.shape[1]-1, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    history = model.fit(xtr, ytr, epochs=50, batch_size=1, verbose=0)

    # Evaluate and Predict
    print(f'For {ii} steps')
    print("==================================")
    scores = model.evaluate(xtr, ytr, verbose=0)  # evaluate the model
    ytr_pred = model.predict(xtr, batch_size=1, verbose=0)  # make predictions
    print("Train MSE: ", scores)
    print("Train R2: ", r2_score(ytr, ytr_pred))
    yte_pred = model.predict(xte, batch_size=1, verbose=0)  # make predictions
    print("Test MSE: ", mean_squared_error(yte, yte_pred))
    print("Test R2: ", r2_score(yte, yte_pred))

    steps = 1
    xtr = np.reshape(xtr, (xtr.shape[0], steps, xtr.shape[1]))
    ytr = np.reshape(ytr, (xtr.shape[0], steps, 1))
    print(xtr.shape, ytr.shape)

    xte = np.reshape(xte, (xte.shape[0], steps, xte.shape[1]))
    yte = np.reshape(yte, (xte.shape[0], steps, 1))
    print(xte.shape, yte.shape)

    # Building a model with SimpleRNN
    batch_size = 1
    model = Sequential()

    model.add(SimpleRNN(units=len(xtr), input_shape=(xtr.shape[1], xtr.shape[2]), activation="relu", return_sequences=True))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    model.fit(xtr, ytr, epochs=50, batch_size=batch_size, verbose=0)

    # Predicting and plotting the result
    trainPredict = model.predict(xtr, batch_size=batch_size)
    testPredict = model.predict(xte, batch_size=batch_size)

    # invert predictions
    trainPredict = np.reshape(trainPredict, (xtr.shape[0]*steps, 1))
    ytr2d = np.reshape(ytr, (xtr.shape[0]*steps, 1))
    testPredict = np.reshape(testPredict, (xte.shape[0]*steps, 1))
    yte2d = np.reshape(yte, (xte.shape[0]*steps, 1))

    # calculate error
    print("Train MSE: ", mean_squared_error(ytr2d, trainPredict))
    print("Train R2: ", r2_score(ytr2d, trainPredict))
    print("Test MSE: ", mean_squared_error(yte2d, testPredict))
    print("Test R2: ", r2_score(yte2d, testPredict))
