"""
The big diseases
'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'
"""

import glob
import pickle
import sys

#box2
import keras
import keras.backend as K

from keras import initializers
from keras.callbacks import ModelCheckpoint
from keras.engine import InputSpec
from keras.layers import Wrapper
from keras.layers import Lambda, Wrapper

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array

import numpy as np
np.random.seed(0)

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

#from concrete_dropout import SpatialConcreteDropout, ConcreteDropout

class SpatialConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given Conv2D input layer.
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, data_format=None, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(SpatialConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(SpatialConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            input_dim = input_shape[1] # we drop only channels
        else:
            input_dim = input_shape[3]
        
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2. / 3.

        input_shape = K.shape(x)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        unif_noise = K.random_uniform(shape=noise_shape)
        
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)



def load_and_pad_image(image_filename, max_image_shape=(450,450,1)):
    location = "/data/chexpert/CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"

    oimage = img_to_array(Image.open(image_filename))

    fimage=np.ones(max_image_shape)
    fimage[:min(oimage.shape[0],fimage.shape[0]),
           :min(oimage.shape[1],fimage.shape[0]),
           :min(oimage.shape[2],fimage.shape[0]),
          ] = oimage[:fimage.shape[0], :fimage.shape[1], :fimage.shape[2]]

    return fimage


def load_chexpert_dataframe(train_or_valid='train'):
    csv_filename="/data/chexpert/CheXpert-v1.0-small/{}.csv".format(train_or_valid)
    return pd.read_csv(csv_filename)

def batchLoader(dataset, batch_size, img_rows=450, img_cols=450):
    L = len(dataset)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = [load_and_pad_image(image_filename) for image_filename, _ in dataset[batch_start:limit]]
            #print('finished X')
            XX = np.asarray(X)
            #print('finished XX')
            
            if K.image_data_format() == 'channels_first':
                XX = XX.reshape(XX.shape[0], 1, img_rows, img_cols)
                #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
                #input_shape = (1, img_rows, img_cols)
            else:
                XX = XX.reshape(XX.shape[0], img_rows, img_cols, 1)
                #x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                #input_shape = (img_rows, img_cols, 1)

            XX = XX.astype('float32')

            #x_train = x_train.astype('float32')
            #x_test = x_test.astype('float32')
            #x_train /= 255
            #x_test /= 255

            #X = someMethodToLoadImages(files[batch_start:limit])

            #Y = someMethodToLoadTargets(files[batch_start:limit])
            Y = [i[1] for i in dataset][batch_start:limit]
            #yy = np.asarray([[i] for i in y])
            YY = np.asarray(Y)

            yield (XX, YY) #a tuple with two numpy arrays with batch_size samples     

            del X, XX

            batch_start += batch_size   
            batch_end += batch_size

base_image_path='/data/chexpert/'

df = load_chexpert_dataframe('train')
disease_columns = list(df.columns)
disease_columns.remove("Path")
disease_columns.remove("Sex")
disease_columns.remove("Age")
disease_columns.remove("Frontal/Lateral")
disease_columns.remove("AP/PA")
disease_columns.remove("No Finding")
#disease_to_consider=disease_columns[int(sys.argv[1])]
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
disease_to_consider = diseases[int(sys.argv[1])]
#disease_to_consider = diseases[0]
print("disease_to_consider={}".format(disease_to_consider))

batch_size = 32
num_classes = 1
epochs = 1

# input image dimensions
img_rows, img_cols = 450, 450

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


all_values = df[['Path',disease_to_consider]].dropna().values#[:128]
dataset = [(base_image_path+i[0],i[1]) for i in all_values if i[1] in [0.0, 1.0]]
train, test = train_test_split(dataset, test_size=0.1, random_state=42)
print('len(train)={}'.format(len(train)))
print('len(test)={}'.format(len(test)))
#image_filenames = [base_image_path+i[0] for i in dataset]

#x = [load_and_pad_image(image_filename) for image_filename, _ in dataset]
#print("completed loading images")
#y = [i[1] for i in dataset]

#xx = np.asarray(x)
#yy = np.asarray([[i] for i in y])
#x_train,x_test,y_train,y_test = train_test_split(xx, yy, random_state=42)
#print("len(ytrain)={}, len(ytest)={}".format(len(ytrain), len(ytest)))
#x_train = xx[:128]
#y_train = yy[:128]
#x_test = xx[128:]
#y_test = yy[128:]

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

N = len(train)
wd = 1e-2 / N
dd = 2. / N
model = Sequential()
model.add(SpatialConcreteDropout(Conv2D(32, kernel_size=(3, 3),activation='relu'),weight_regularizer=wd, dropout_regularizer=dd,input_shape=input_shape))
model.add(SpatialConcreteDropout(Conv2D(64, (3, 3), activation='relu'),
                                 weight_regularizer=wd, dropout_regularizer=dd))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(ConcreteDropout(Dense(128, activation='relu'), 
                          weight_regularizer=wd, dropout_regularizer=dd))
model.add(ConcreteDropout(Dense(num_classes, activation='softmax'), 
                          weight_regularizer=wd, dropout_regularizer=dd))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model_name = 'saved_models/{}'.format(disease_to_consider.replace(" ","")) + "_epoch_{epoch:02d}_val_acc_{val_acc:.2f}.h5"
#checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=False)
#callbacks_list = [checkpoint]

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1)
#          validation_data=(x_test, y_test),
#          callbacks=callbacks_list)
steps_per_epoch = int(len(train) / batch_size)
model.fit_generator(batchLoader(dataset, batch_size), steps_per_epoch, epochs = 1, verbose=2)
print('completed training')
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#box 16
ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
print(ps)

model_name = "saved_models/{}".format(disease_to_consider.replace(" ",""))
model.save(model_name+".h5")
print("saved model to disk as h5 file: {}".format(model_name+".h5"))
#with open(model_name+".pickle", "wb") as gh:
#    gh.write(pickle.dump(model))
#print("saved model to disk as pickle file: {}".format(model_name+".pickle"))
