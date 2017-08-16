import numpy as np
import os
import glob
from scipy import misc
from six.moves import cPickle as pickle
import keras
import string
import h5py


shape = (50, 120)
characters = string.digits + string.ascii_lowercase
n_class = len(characters)

def load_data():
	datafile = "dataset.pickle"
	if not os.path.exists(datafile):
		samples = glob.glob("sample/*.jpg")
		total = len(samples)
		nb_train = int(0.9*total)
		datasets = np.zeros((total,50,120,3))
		labels = [np.zeros((total, n_class)) for i in range(4)] 
		np.random.shuffle(samples)
		for i in range(len(samples)):
			x = misc.imresize(misc.imread(samples[i]),shape)
			#print(x.shape)
			datasets[i] = x
			strs = samples[i].split("/")[-1].split(".")[0]
			for j, ch in enumerate(strs):
				print(characters,ch,characters.find(ch))
				labels[j][i][characters.find(ch)] = 1

		train_dataset = datasets[:nb_train,:,:,:]
		train_labels = []
		for i in range(4):
			train_labels.append(labels[i][:nb_train,:])
		test_dataset = datasets[nb_train:,:,:,:]
		test_labels = []
		for i in range(4):
			test_labels.append(labels[i][nb_train:,:])

		with open(datafile, "wb") as f:
			pickle.dump({"train_dataset":train_dataset, "train_labels":train_labels, "test_dataset":test_dataset, "test_labels":test_labels}, f, pickle.HIGHEST_PROTOCOL)
	with open(datafile, "rb") as f:
		data = pickle.load(f)
	return data["train_dataset"], data["train_labels"], data["test_dataset"], data["test_labels"]
def decode(y):
	num = y[0].shape[0]
	chars = []
	for j in range(num):
		chars.append("".join([characters[np.argmax(y[i][j], axis=0)] for i in range(len(y))]))
	return chars
		

train_dataset, train_labels, test_dataset, test_labels  = load_data()

#construct model
input_tensor = keras.layers.Input(shape=(50, 120,3))
x = input_tensor
for i in range(3):
	x = keras.layers.Conv2D(32*2**i, 3, padding="valid", activation="relu")(x)
	x = keras.layers.Conv2D(32*2**i, 3, padding="valid", activation="relu")(x)
	x = keras.layers.pooling.MaxPool2D((2,2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(0.25)(x)
x = [keras.layers.Dense(n_class, activation="softmax", name="c%d"%(i+1))(x) for i in range(4)]
model = keras.models.Model(input=input_tensor, output=x)
model.compile(optimizer="Adadelta",loss="categorical_crossentropy")

#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file="model.png", show_shapes=True)

model.fit(train_dataset, train_labels, batch_size=128, epochs=50,validation_split=0.2)
#model.load_weights("model.h5")
y_hat = model.predict(test_dataset)
print(y_hat[0])
print(test_dataset.shape)
print(y_hat[0].shape)
char_pre = decode(y_hat)
char_rel = decode(test_labels)
print(list(zip(char_rel, char_pre)))
acc = sum([1 for i in range(len(char_pre)) if char_pre[i]==char_rel[i]])/len(char_pre)
print(acc)
model.save_weights("model.h5")1,1           Top
