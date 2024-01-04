# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import gc

import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD



def model_params():
    
    # define parameters for model, residual blocks, etc
    # returns dictionarty 'params'
    
    #load cifar10 data for this example
    (x_train,y_train),(_,_) = cifar10.load_data()
    
    examples,width, height, channels = np.shape(x_train)
    
    batch_size = 256
    num_classes = 10
    validation_split = 0.2
    
    #define number of residual layer stacks
    #each residual layer stack contains 1 convolutonal block + 1 or more identity block      
    stack = 4
    
    # residual layer stack type: 'fixed' or 'cascading'
    # 'fixed' means one residual layer stack has exactly 1 convolutional block and 1 identity block
    # 'cascading' means one residual layer stack has 1 convolutional block + 'stack number' of identity blocks
    # e.g stack 1 has 1 identiy block, stack 2 has 2 identity block etc
    
    # block_type = 'fixed'
    res_stack_type = 'cascading'
    
    init_filter = 64
    
    # number of filters for each convolutional step in both convolutional and identity block
    # can be same or different
    filters = np.array( [64,64,64])
    epochs = 100
    
    #loss, optimizer, metrics definition
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=None)        
    
    metrics = ['accuracy']
    learning_rate = 0.1
    optimizer_momentum = 0.9
    optimizer = SGD(learning_rate = learning_rate, momentum = optimizer_momentum)
    
    #dictionary of all params
    params = {
        'width':width,
        'height':height,
        'channels':channels,
        'batch_size':batch_size,
        'classes':num_classes,
        'val_split':validation_split,
        'num_stack':stack,
        'res_stack_type':res_stack_type,
        'filters':filters,
        'init_filter':init_filter,
        'epochs':epochs,
        'loss':loss,
        'initializer':initializer,
        'optimizer':optimizer,
        'metrics':metrics
        }
    
    return params


def preprocess_data():
    
    #preprocess data for image datagenerator based data augmentation
    #split data into train, validation & test batches
    # returns train, validation & test batches
    
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    params = model_params()
    y_train = to_categorical(y_train,params.get('classes'))
    y_test = to_categorical(y_test,params.get('classes'))

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                        validation_split= params.get('val_split'),
                        preprocessing_function = tf.keras.applications.resnet50.preprocess_input,
                        horizontal_flip = True,
                        rescale = 1./255,
                        rotation_range = 30,
                        width_shift_range = 0.2,
                        height_shift_range = 0.2,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        fill_mode = 'nearest'   
                        )
        
    train_batches = train_generator.flow(x_train,y_train,batch_size = params.get('batch_size'),subset = 'training')
   
    validation_batches = train_generator.flow(x_train,y_train,batch_size = params.get('batch_size'),subset='validation') 
    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function= tf.keras.applications.resnet50.preprocess_input,
                        rescale = 1./255
                        )
    
    test_batches = test_generator.flow(x_test,y_test,batch_size= params.get('batch_size'))
        
    return train_batches, validation_batches,test_batches


def identity_block(x, num_filters):
    # identity block of residual layer stack
    #inputs: x - data tensor, num_filters- filters for each convolutional layer
    #reurns x 
    
    params = model_params()
    init = params.get('initializer')
    x_skip = x
    f1,f2,f3 = num_filters

    # main path of residual identity block
    x = tfl.Conv2D(filters = f1,kernel_size = (1,1),strides = (1,1),padding = 'valid',kernel_initializer = init)(x)  
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation('relu')(x)
    
    x = tfl.Conv2D(filters = f2,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = init)(x)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation('relu')(x)
    
    x = tfl.Conv2D(filters = f3,kernel_size = (1,1),strides = (1,1),padding = 'valid',kernel_initializer = init)(x)
    x = tfl.BatchNormalization(axis=3)(x)      
    
    # adding the skip connection
    x = tfl.Add()([x,x_skip])
    x = tfl.Activation('relu')(x)    

    
    return x


def convolutional_block(x, num_filters):
    # convolutional block of residual layer stack
    #inputs: x - data tensor, num_filters- filters for each convolutional layer
    #reurns x 
    
    params = model_params()
    init = params.get('initializer')
    x_skip = x
    f1,f2,f3 = num_filters
    _,w,h,c = np.shape(x)    
   
    # dimensionality reduction happens in this block, check for zero dimension errors
    if (w<4 or h<4):
        return x
   
    
   # main path of convolutional residual block
    x = tfl.Conv2D(filters = f1,kernel_size = (1,1),strides = (2,2),padding = 'valid',kernel_initializer = init)(x)  
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation('relu')(x)
    
    x = tfl.Conv2D(filters = f2,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = init)(x)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation('relu')(x)
    
    x = tfl.Conv2D(filters = f3,kernel_size = (1,1),strides = (1,1),padding = 'valid',kernel_initializer = init)(x)
    x = tfl.BatchNormalization(axis=3)(x)
    
    # matching dimensions for skip connection
    x_skip = tfl.Conv2D(filters = f3,kernel_size = (1, 1),strides = (2,2),padding = 'valid',kernel_initializer = init)(x_skip)
    x_skip = tfl.BatchNormalization(axis=3)(x_skip)    
    
    # adding the skip connection
    x = tfl.Add()([x,x_skip])
    x = tfl.Activation('relu')(x)    

    return x


def Res_layerstack(x):
    # residual layer stack definition
    #inputs x
    #returns x
    
    params = model_params()
    num_stacks = params.get('num_stack')
    res_stack_type = params.get('res_stack_type')
    num_filters = params.get('filters')
    
    # number of residual layer stacks
    for stage in range(num_stacks):
        
        # double number of filters after every residual layer stack
        if (stage >0):
            num_filters *= 2
            
        # data progerssion through residual layer stack for 'fixed' & 'cascading' stack type
        if (res_stack_type == 'fixed'):
           # 'fixed' stack type is static with 1 convolutional block and 1 identity block for all stacks
            x = convolutional_block(x,num_filters)
            x = identity_block(x,num_filters)
            
        elif (res_stack_type== 'cascading'):
            #'cascading' stack type is dynamic with 1 convolutional block and 'stack number' of identity blocks
            x = convolutional_block(x, num_filters)            
            for i in range(stage+1):
                x = identity_block(x,num_filters)                 
                        
    return x


def create_model(inputs):
    #model defintiion with params
    # input - inputs in tensor form or data generator
    # returns model
    
    params = model_params()
    initializer = params.get('initializer')
    num_classes = params.get('classes')
    width = params.get('width')
    height = params.get('height')
    channels = params.get('channels')
    init_filter = params.get('init_filter')
    inputs= tfl.Input(shape=(width,height,channels))
    loss = params.get('loss')
    optimizer = params.get('optimizer')
    metrics = params.get('metrics')
    
    # first convolution layer before residual layer stack
    x = tfl.Conv2D(filters = init_filter,kernel_size = (3,3),strides = (1,1),padding = 'valid',kernel_initializer = initializer)(inputs)  
    x = tfl.BatchNormalization()(x)
    x = tfl.Activation('relu')(x)
    
    # data progression into residual layer stack    
    x = Res_layerstack(x)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.5)(x)
    x = tfl.Flatten()(x)
    
    # Fully connected output layer
    outputs = tfl.Dense(num_classes,activation='softmax',kernel_initializer = initializer)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss = loss,optimizer = optimizer,metrics = metrics)
    model.summary()
    
    return model



#%%

gc.enable()

# get params
params = model_params()

# preproess data
train_batches, validation_batches,test_batches = preprocess_data()

# create model
model = create_model(train_batches)

#train model
history = model.fit(train_batches,
                    batch_size=params.get('batch_size'),
                    epochs = params.get('epochs'),
                    verbose = 1,
                    validation_data= validation_batches )

# quick test of model performance
score = model.evaluate(test_batches, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
# model.save('D:/Analysis/cascaderesnet/traffic.h5')

#%%
# visualize plots of training & validation loss and accuracy

acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]
epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc, label = 'Train Acc' )
plt.plot  ( epochs, val_acc, label = 'Val Acc' )
plt.title ('Training and validation accuracy')
plt.xlabel ('epochs')
plt.legend(loc="upper right")
fig1 = plt.gcf()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss,label = 'Train loss' )
plt.plot  ( epochs, val_loss , label = 'Val loss')
plt.title ('Training and validation loss'   )
plt.xlabel ('epochs')
plt.legend(loc="lower right")
fig2 = plt.gcf()

classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_test = x_test.astype('float32')/255

# generate precision, recall classification report for all classes
predictions = model.predict(x_test,batch_size=256)
pred_report = classification_report(y_test,predictions.argmax(axis=1), target_names=classes)
print(pred_report)

del model

gc.collect()

