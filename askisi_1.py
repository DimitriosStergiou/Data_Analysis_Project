import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_excel('/Users/johnmakris/Downloads/CTG.xls', sheet_name='Data', header=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.shape)
df = df[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
         'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'CLASS', 'NSP']]


# delete Nan values
df = df.dropna()

# PCA
df_standard = pd.DataFrame(StandardScaler().fit_transform(df.iloc[:, 0:-2].values), columns=df.columns[:-2])
df_standard['CLASS'] = df['CLASS']
df_standard['NSP'] = df['NSP'] - 1
num_of_columns = [ii for ii in range(1, len(df.columns) - 4)]
variance_explained = []
for ii in num_of_columns:
    pca = PCA(n_components=ii)
    pca.fit(df_standard.iloc[:, 0:-2].T)
    variance_explained.append(sum(pca.explained_variance_ratio_))

# plot explained variance
fig, ax = plt.subplots()
ax.plot(num_of_columns, variance_explained)
ax.set_title('Plot of explained variance from PCA', fontsize=14, fontweight='bold')
ax.set_xticks(num_of_columns)
ax.set_xlabel('Number of features', fontweight='bold')
ax.set_ylabel('Variance ratio explained', fontweight='bold')
plt.show()
fig.savefig('/Users/johnmakris/Downloads/plot_pca.png', bbox_inches='tight')

pca = PCA(n_components=10)
pca.fit(df_standard.iloc[:, 0:-2].T)
print(f'Variance explained using 13 components: {sum(pca.explained_variance_ratio_)}')
df_pca = pd.DataFrame(pca.components_).T


#clustering

# KMeans with PCA dataframe
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_pca)
print(accuracy_score(df_standard['NSP'].values, kmeans.labels_))
print(confusion_matrix(df_standard['NSP'].values, kmeans.labels_))

# KMeans with standardized dataframe
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_standard.iloc[:, 0:-2].values)
print(accuracy_score(df_standard['NSP'].values, kmeans.labels_))
print(confusion_matrix(df_standard['NSP'].values, kmeans.labels_))

fig = plt.figure()  # make 3d fig
ax = Axes3D(fig)
ax.scatter(df_pca.iloc[:, 3], df_pca.iloc[:, 0], df_pca.iloc[:, 2], c=df_standard['NSP'].values, edgecolor='k')
fig.show()


dbscan = DBSCAN(eps=0.2, min_samples=10)  # define the model
dbscan = dbscan.fit(df_pca)  # fit the model
labels = dbscan.labels_
print(np.unique(labels +1))
print(confusion_matrix(df_standard['NSP'].values, labels+1))
print(accuracy_score(df_standard['NSP'].values, labels+1))


optics = OPTICS(min_samples=14, max_eps=3)  # define the model
optics = optics.fit(df_standard.iloc[:, 0:-2].values)  # fit the model
labels = optics.labels_
print(np.unique(labels))
print(confusion_matrix(df_standard['NSP'].values, labels+1))
print(accuracy_score(df_standard['NSP'].values, labels+1))

optics = OPTICS(min_samples=15, max_eps=3)  # define the model
optics = optics.fit(df_pca.values)  # fit the model
labels = optics.labels_
print(np.unique(labels))
print(confusion_matrix(df_standard['NSP'].values, labels+1))
print(accuracy_score(df_standard['NSP'].values, labels+1))


# classification with MLP
data = shuffle(df_standard.values)
X = data[:, :-2]
y = data[:, -2]
encoder = LabelEncoder().fit(y)
y_bool = encoder.transform(y)
y = np_utils.to_categorical(y_bool)

len_data = df_standard.iloc[:, 0:-2].shape[0]
print(len_data)
train_size = int(len_data * .6)
valid_size = int(len_data * .2)
print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (len_data - (train_size+valid_size)))

xtr = X[:train_size, :]
ytr = y[:train_size, :]
ytr_bool = y_bool[:train_size]

xva = X[train_size:train_size+valid_size, :]
yva = y[train_size:train_size+valid_size, :]
yva_bool = y_bool[train_size:train_size+valid_size]

xte = X[train_size+valid_size:, :]
yte = y[train_size+valid_size:, :]
yte_bool = y_bool[train_size+valid_size:]

# keras model
model = Sequential()
model.add(Dense(21, input_dim=21, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=50, batch_size=1, verbose=0)

print(history.history)
# Plot training and validation loss
fig = plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(2, 1, 2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
fig.savefig('/Users/johnmakris/Downloads/keras_10_10.png', bbox_inches='tight')

# Evaluate and Predict
scores = model.evaluate(xtr, ytr, verbose=0)
print("Train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

ytr_pred = model.predict_classes(xtr, verbose=0)
print("Train Accuracy by model.predict: %.2f%%" % (100*sum(ytr_bool == ytr_pred)/ytr.shape[0]))

# make class predictions with the model
yva_pred = model.predict_classes(xva, verbose=0)
print("Val Accuracy by model.predict: %.2f%%" % (100*sum(yva_bool == yva_pred)/yva.shape[0]))

# make class predictions with the model
yte_pred = model.predict(xte, batch_size=1, verbose=0)
yte_pred_bool = np.argmax(yte_pred, axis=1)

print("Test Accuracy by model.predict: %.2f%%" % (100*sum(yte_bool == yte_pred_bool)/yte.shape[0]))

model = Sequential()
model.add(Dense(21, input_dim=21, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=50, batch_size=1, verbose=0)

print(history.history)
# Plot training and validation loss
fig = plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(2, 1, 2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
fig.savefig('/Users/johnmakris/Downloads/keras_50_50.png', bbox_inches='tight')

# Evaluate and Predict
scores = model.evaluate(xtr, ytr, verbose=0)
print("Train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

ytr_pred = model.predict_classes(xtr, verbose=0)
print("Train Accuracy by model.predict: %.2f%%" % (100*sum(ytr_bool == ytr_pred)/ytr.shape[0]))

# make class predictions with the model
yva_pred = model.predict_classes(xva, verbose=0)
print("Val Accuracy by model.predict: %.2f%%" % (100*sum(yva_bool == yva_pred)/yva.shape[0]))

# make class predictions with the model
yte_pred = model.predict(xte, batch_size=1, verbose=0)
yte_pred_bool = np.argmax(yte_pred, axis=1)

print("Test Accuracy by model.predict: %.2f%%" % (100*sum(yte_bool == yte_pred_bool)/yte.shape[0]))


model = Sequential()
model.add(Dense(21, input_dim=21, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=50, batch_size=1, verbose=0)

print(history.history)
# Plot training and validation loss
fig = plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(2, 1, 2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
fig.savefig('/Users/johnmakris/Downloads/keras_100_100.png', bbox_inches='tight')

# Evaluate and Predict
scores = model.evaluate(xtr, ytr, verbose=0)
print("Train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

ytr_pred = model.predict_classes(xtr, verbose=0)
print("Train Accuracy by model.predict: %.2f%%" % (100*sum(ytr_bool == ytr_pred)/ytr.shape[0]))

# make class predictions with the model
yva_pred = model.predict_classes(xva, verbose=0)
print("Val Accuracy by model.predict: %.2f%%" % (100*sum(yva_bool == yva_pred)/yva.shape[0]))

# make class predictions with the model
yte_pred = model.predict(xte, batch_size=1, verbose=0)
yte_pred_bool = np.argmax(yte_pred, axis=1)

print("Test Accuracy by model.predict: %.2f%%" % (100*sum(yte_bool == yte_pred_bool)/yte.shape[0]))
