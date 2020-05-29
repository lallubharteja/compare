# Fix the random seed
from numpy.random import seed
#seed(1)
from tensorflow import random
#random.set_seed(2) 

# Load various imports
import pandas as pd
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from datetime import datetime
import argparse

def non_nan_average(x):
    # Computes the average of all elements that are not NaN in a rank 1 tensor
    nan_mask = tf.math.is_nan(x)
    x = tf.boolean_mask(x, tf.logical_not(nan_mask))
    return K.mean(x)


def uar_accuracy(y_true, y_pred):
    # Calculate the label from one-hot encoding
    pred_class_label = K.argmax(y_pred, axis=-1)
    true_class_label = K.argmax(y_true, axis=-1)

    cf_mat = tf.math.confusion_matrix(true_class_label, pred_class_label )

    diag = tf.linalg.tensor_diag_part(cf_mat)    

    # Calculate the total number of data examples for each class
    total_per_class = tf.reduce_sum(cf_mat, axis=1)

    acc_per_class = diag / tf.maximum(1, total_per_class)  
    uar = non_nan_average(acc_per_class)

    return uar

def standardize(X):
    mean = np.mean(X, axis=(0,2))
    mean = mean.reshape((1, X.shape[1], 1,1))
    
    std = np.std(X, axis=(0,2))
    std = std.reshape((1, X.shape[1], 1,1))

    return (X-mean)/std
   
def extract_features(file_name):   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled

parser = argparse.ArgumentParser(description='Run the seq2seq model.')
parser.add_argument('--taskname',
        type=str,
        default='ComParE2020_Mask',
        help="set the task name")
parser.add_argument('--subid',
        type=str,
        default='1',
        help="set the submission id")
parser.add_argument('--extralayers',
        type=int,
        default=0,
        help="set the number of extra layers (default 0)")
parser.add_argument('--keep',
        type=int,
        default=-1,
        help="set number of rows of features to keep (default -1: keeps everything)")
parser.add_argument('--features',
        type=str,
        default='ComParE',
        help="set the feature set to be used for classification")
parser.add_argument('--predict',
        action='store_const',
        const=True,
        default=False,
        help="pass this flag to run predictions on the test set, make sure you have the feature file stored at generated/features/")
parser.add_argument('--devpredict',
        action='store_const',
        const=True,
        default=False,
        help="pass this flag to run predictions on the test set, make sure you have the feature file stored at generated/features/")
parser.add_argument('--standardize',
        action='store_const',
        const=True,
        default=False,
        help="pass this flag to run predictions on the test set, make sure you have the feature file stored at generated/features/")
args = vars(parser.parse_args())

# Task
task_name  = args['taskname']  # os.getcwd().split('/')[-2]

# Enter your team name HERE
team_name = 'Aalto'

# Enter your submission number HERE
submission_index = args['subid']

# Configuration
feature_set = args['features']

feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
             'BoAW-125':     ( 250, 1, ';',  None),
             'BoAW-250':     ( 500, 1, ';',  None),
             'BoAW-500':     (1000, 1, ';',  None),
             'BoAW-1000':    (2000, 1, ';',  None),
             'BoAW-2000':    (4000, 1, ';',  None),
             'auDeep-30':    (1024, 2, ',', 'infer'),
             'auDeep-45':    (1024, 2, ',', 'infer'),
             'auDeep-60':    (1024, 2, ',', 'infer'),
             'auDeep-75':    (1024, 2, ',', 'infer'),
             'auDeep-fused': (4096, 2, ',', 'infer'),
             'mfcc': (3920, 1, ';', 'infer'),
             'mel40.wl160.preemp0': (6920, 1, ';', 'infer'),
             'mel40.wl160.preemp0.97': (6920, 1, ';', 'infer'),
             'mel100.wl160.preemp0':(17300, 1, ';', 'infer'),
             'mel100.wl480.preemp0':(17300, 1, ';', 'infer'),
             'mel100.wl800.preemp0':(17300, 1, ';', 'infer'),
             'mel10.wl160.preemp0':(1730, 1, ';', 'infer'),
             'mel10.wl480.preemp0':(1730, 1, ';', 'infer'),
             'mel10.wl800.preemp0':(1730, 1, ';', 'infer'),
             'mel20.wl160.preemp0':(3460, 1, ';', 'infer'),
             'mel20.wl480.preemp0':(3460, 1, ';', 'infer'),
             'mel20.wl800.preemp0':(3460, 1, ';', 'infer'),
             'mel40.wl160.preemp0':(6920, 1, ';', 'infer'),
             'mel40.wl480.preemp0':(6920, 1, ';', 'infer'),
             'mel40.wl800.preemp0':(6920, 1, ';', 'infer'),
             'mel50.wl160.preemp0':(8650, 1, ';', 'infer'),
             'mel50.wl480.preemp0':(8650, 1, ';', 'infer'),
             'mel50.wl800.preemp0':(8650, 1, ';', 'infer'),
             'mel200.wl160.preemp0':(34600, 1, ';', 'infer'),
             'mel200.wl480.preemp0':(34600, 1, ';', 'infer'),
             'mel200.wl800.preemp0':(34600, 1, ';', 'infer'),
             'mel200.wl160.preemp0.f400':(34600, 1, ';', 'infer'),
             'mel200.wl480.preemp0.f400':(34600, 1, ';', 'infer'),
             'mel200.wl800.preemp0.f400':(34600, 1, ';', 'infer'),
             'mel200.wl160.preemp-1.f400':(34600, 1, ';', 'infer'),
             'mel200.wl480.preemp-1.f400':(34600, 1, ';', 'infer'),
             'mel200.wl800.preemp-1.f400':(34600, 1, ';', 'infer'),
             'mel200.wl160.preemp-1':(34600, 1, ';', 'infer'),
             'mel200.wl480.preemp-1':(34600, 1, ';', 'infer'),
             'mel200.wl800.preemp-1':(34600, 1, ';', 'infer'),
             'mel200.wl160.preemp-0.97.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl480.preemp-0.97.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl800.preemp-0.97.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl160.preemp0.hmel20':(3460, 1, ';', 'infer'),
             'mel200.wl480.preemp0.hmel20':(3460, 1, ';', 'infer'),
             'mel200.wl800.preemp0.hmel20':(3460, 1, ';', 'infer'),
             'mel200.wl160.preemp0.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl480.preemp0.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl800.preemp0.lmel20':(3460, 1, ';', 'infer'),
             'mel200.wl160.preemp0.lmel10':(1730, 1, ';', 'infer'),
             'mel200.wl480.preemp0.lmel10':(1730, 1, ';', 'infer'),
             'mel200.wl800.preemp0.lmel10':(1730, 1, ';', 'infer'),
             'mel200.wl160.preemp0.hmel20.lmel20':(6920, 1, ';', 'infer'),
             'mel200.wl480.preemp0.hmel20.lmel20':(6920, 1, ';', 'infer'),
             'mel200.wl800.preemp0.hmel20.lmel20':(6920, 1, ';', 'infer'),
             'warped_mfcc': (3920, 1, ';', 'infer'),
             'librosa_mfcc': (6920, 1, ';', 'infer'),
             'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')}

num_feat = feat_conf[feature_set][0]
ind_off  = feat_conf[feature_set][1]
sep      = feat_conf[feature_set][2]
header   = feat_conf[feature_set][3]

# Path of the features and labels
features_path = 'generated/features/'
label_file    = 'data/lab/labels.csv'

# Start
print('\nRunning ' + task_name + ' ' + feature_set + ' seq2seq ... (this might take a while) \n')

# Load features and labels
x_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
x_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
x_test = None
x_traindevel = None
if args['predict']:
    x_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    x_traindevel = np.concatenate((x_train, x_devel))

df_labels = pd.read_csv(label_file)
Y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
Y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
Y_traindevel = None
if args['predict']:
    Y_traindevel = np.concatenate((Y_train, Y_devel))

print('Finished feature extraction from', x_train.shape[0], 'training examples and', x_devel.shape[0], 'development examples.')

# Encode the classification labels
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(Y_train))
y_devel = to_categorical(le.transform(Y_devel))
y_traindevel = None
if args['predict']:
    y_traindevel = to_categorical(le.transform(Y_traindevel))

x_train = x_train.reshape(x_train.shape[0], num_feat, 1)
x_devel = x_devel.reshape(x_devel.shape[0], num_feat, 1)
if args['predict']:
    x_test = x_test.reshape(x_test.shape[0], num_feat, 1)
    x_traindevel = x_traindevel.reshape(x_traindevel.shape[0], num_feat, 1)
    

num_labels = le.classes_.shape[0]

num_columns = 173
num_rows = int(num_feat/num_columns)

num_channels = 1
filter_size = 2

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_devel = x_devel.reshape(x_devel.shape[0], num_rows, num_columns, num_channels)
if args['predict']:
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
    x_traindevel = x_traindevel.reshape(x_traindevel.shape[0], num_rows, num_columns, num_channels)

if args['standardize']:
    # normalize
    x_train = standardize(x_train)
    x_devel =  standardize(x_devel)
    if args['predict']:
        x_traindevel =  standardize(x_traindevel)
        x_test = standardize(x_test)


if args['keep'] > 0: 
    x_train = x_train[:,:args['keep'],:,:]
    x_devel = x_devel[:,:args['keep'],:,:]
    if args['predict']:
        x_traindevel = x_traindevel[:,:args['keep'],:,:]
        x_test = x_test[:,:args['keep'],:,:]
    num_rows = args['keep']

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

if args['extralayers'] > 0:
    filters=64
    for i in range(args['extralayers']):
        model.add(Conv2D(filters=filters, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        filters *= 2

model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=[uar_accuracy], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_devel, y_devel, verbose=1)
accuracy = 100*score[1]

print("Pre-training UAR: %.4f%%" % accuracy) 
num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='generated/saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

if args['predict']: 
    model.fit(x_traindevel, y_traindevel, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_devel, y_devel), callbacks=[checkpointer], verbose=1)
else:
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_devel, y_devel), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing seti
model.load_weights('generated/saved_models/weights.best.basic_cnn.hdf5')
score = model.evaluate(x_devel, y_devel, verbose=0)
print("Testing UAR: ", score[1])

if args['devpredict']:
    y_pred = model.predict(x_devel)
    pred = list(le.inverse_transform(y_pred.argmax(axis=-1)))
    
    # Write out predictions to csv file (official submission format)
    pred_file_name = 'experiments/seq2seq2d/' + task_name + '.' + feature_set + '.keep' + str(num_rows) +'.devel.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('devel')].values,
                            'prediction': pred,
                            'prob_'+le.classes_[0]: y_pred[:,0],
                            'prob_'+le.classes_[1]: y_pred[:,1]},
                      columns=['file_name','prediction','prob_'+le.classes_[0], 'prob_'+le.classes_[1]])
    df.to_csv(pred_file_name, index=False)

if args['predict']:
    y_pred = model.predict(x_test)
    pred = list(le.inverse_transform(y_pred.argmax(axis=-1)))
    
    # Write out predictions to csv file (official submission format)
    pred_file_name = 'experiments/seq2seq2d/' + task_name + '.' + feature_set + '.keep' + str(num_rows) +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': pred,
                            'prob_'+le.classes_[0]: y_pred[:,0],
                            'prob_'+le.classes_[1]: y_pred[:,1]},
                      columns=['file_name','prediction','prob_'+le.classes_[0], 'prob_'+le.classes_[1]])
    df.to_csv(pred_file_name, index=False)
