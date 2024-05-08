  
from keras.datasets import cifar10

  
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

  
print('Traning data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)

  
y_train.shape,y_test.shape

  
import numpy as np
# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

  
import matplotlib.pyplot as plt
%matplotlib inline

  
label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
}

  
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(x_train[0], (32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_train[0][0]]) + ")"))

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(x_test[0],(32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_test[0][0]]) + ")"))

  
np.min(x_train),np.max(x_train)

  
x_train = x_train/255.0

  
np.min(x_train),np.max(x_train)

  
x_train.shape

  
x_train_flat = x_train.reshape(-1,3072)

  
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]

  
import pandas as pd
df_cifar = pd.DataFrame(x_train_flat,columns=feat_cols)

  
df_cifar['label'] = y_train
print('Size of the dataframe: {}'.format(df_cifar.shape))

  
df_cifar.head()

  
from sklearn.decomposition import PCA

pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:,:-1])

  
principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar
             , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train

  
principal_cifar_Df.head()

  
print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))


import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_cifar_Df,
    legend="full",
    alpha=0.3
)

  
x_test = x_test/255.0

  
x_test = x_test.reshape(-1,32,32,3)

  
x_test_flat = x_test.reshape(-1,3072)

  
pca = PCA(0.9)

  
pca.fit(x_train_flat)

  
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

  
pca.n_components_

  
train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)

  
from keras.models import Sequential
from keras.layers import Dense
# from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import RMSprop

  
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

  
batch_size = 128
num_classes = 10
epochs = 20

  
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(99,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

  
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(test_img_pca, y_test))

  
# Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='acc', color='red')
plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

  
from keras.preprocessing import image
# label mapping
labels = '''airplane automobile bird cat deer dog frog horseship truck'''.split()

# select the image from our test dataset
# image_number = 0
image_path = "imageTest/kucing.jpg"
img = image.load_img(image_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = img_array.reshape(-1, 3072)
img_array = img_array / 255.0
plt.imshow(img_array.reshape((32, 32, 3)))

# display the image
# plt.imshow(x_test[image_number])
# plt.imshow(x_test[image_number])

# load the image in an array
# n = np.array(x_test[image_number])

# reshape it
# p = n.reshape(-1, 3072)

# pass in the network for prediction and 
# save the predicted label label_dict
# predicted_label = labels[model.predict(p).argmax()]
predicted_label = labels[model.predict(img_array).argmax()]

# load the original label
# original_label = labels[np.argmax(img_array)]

# display the result
# print("Original label is {} and predicted label is {}".format(
	# original_label, predicted_label))
print("Predicted label is {}".format(predicted_label))

  
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(x_test_flat, y_test))

  
# Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='acc', color='red')
plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
plt.legend()


