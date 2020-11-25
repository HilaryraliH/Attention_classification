# coding=utf-8
import keras.backend as K
from keras.layers import *
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.engine.topology import Layer
from keras.callbacks import LearningRateScheduler

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_mdl_file(cfg):
    # 若只有一个模型cfg，则直接load并返回即可
    mdl = load_model(cfg.save_mdl_file, custom_objects={'square': square, 'log': log})

    return mdl


def blt_mdl(cfg,cfg_2=None):
    '''
    按照特定的输入形式，建立模型
    注意，最终建立的模型，还是靠cfg来控制，cfg_2中的模型设置是摆设
    '''
    model = None
    if cfg_2:
        model_input1 = get_mdl_in_shape(cfg)
        model_input2 = get_mdl_in_shape(cfg_2)
        model_input = [model_input1,model_input2]
    else:
        model_input = get_mdl_in_shape(cfg)
    model = eval(cfg.mdl_nm)(model_input,cfg)
    model.compile(loss='binary_crossentropy', optimizer=cfg.optimizer, metrics=['accuracy'])
    return model


def blt_bnch_mdl(cfg_1,cfg_2):
    '''
    :param cfg_1: 模型1的参数
    :param cfg_2: 模型2的参数
    :return: 两个模型拉通作为分支（两个输入，结尾时合并为一个输出）
    '''
    mdl_in1 = get_mdl_in_shape(cfg_1)
    mdl_in2 = get_mdl_in_shape(cfg_2)
    mdl1 = load_model(cfg_1.save_mdl_file, custom_objects={'square': square, 'log': log})
    mdl2 = load_model(cfg_2.save_mdl_file, custom_objects={'square': square, 'log': log})

    # 冻结模型
    for i in range(len(mdl1.layers)):
        mdl1.layers[i].trainable = False
    for j in range(len(mdl2.layers)):
        mdl2.layers[j].trainable = False

    # 模型连接处的输出
    out1 = mdl1(mdl_in1)
    out2 = mdl2(mdl_in2)

    # 以相加的方式相连
    out_add = Add()([out1, out2])
    pre = Dense(2, activation='softmax')(out_add)

    mdl = Model([mdl_in1,mdl_in2], pre)
    return mdl


def get_mdl_in_shape(cfg):
    '''
    按照cfg里面的设置，进行相应的数据输入格式设置，即初始化 model_input 的形状
    '''
    if cfg.dt_fm == '2D':
        model_input = Input(shape=(len(cfg.elec), cfg.sam_pnts, 1),name='input1') # chans, points, 1
    elif cfg.dt_fm == '3D':
        model_input = Input(shape=(9,9, cfg.sam_pnts, 1),name='input2') # 空间二位矩阵, points, 1
    return model_input


def eval_mdl(model, X_test, Y_test):
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    true_label = Y_test.argmax(axis=-1)
    acc = np.mean(preds == true_label)
    confu_mat = confusion_matrix(true_label, preds, labels=[0, 1])
    return acc, confu_mat


def fit_mdl(model, cfg, X_train, Y_train, X_test, Y_test):
    schedule = StepDecay(initAlpha=0.001,factor=0.4,dropEvery=10) # 构造阶梯型学习率衰减
    schedule.plot([i for i in range(30)],cfg)
    call_backs = [LearningRateScheduler(schedule)]
    hist = model.fit(X_train, Y_train,
                     batch_size=cfg.batch_size, 
                     epochs=cfg.epochs, 
                     verbose=2, 
                     callbacks=call_backs,
                     shuffle = True,
                     validation_data=(X_test, Y_test))
    model.save(cfg.save_mdl_file)
    return hist



# 在 fit_model 里面用到，用于调整学习率
class LearningRateDecay:
    def plot(self, epochs,cfg, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]

        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.savefig(cfg.lr_change_file)
        plt.close()

# 在 fit_model 里面用到，用于调整学习率
class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.001, factor=0.4,dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)


'''  #######################################  以下是2D模型的结构  ####################################### '''

def JnecnnOrigin(model_input,cfg, nb_classes=2):
    # article: Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI
    # # remain unchanged
    data = Permute((3,2,1))(model_input)
    data = Conv2D(20, (1, 4), strides=(1, 2), activation='relu')(data)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = Conv2D(40, (1, 3), activation='relu')(data)
    data = Conv2D(60, (1, 2), activation='relu')(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(100, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    return model


def JnecnnOriginConstraint(model_input,cfg, nb_classes=2):
    # article: Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI
    # # remain unchanged
    data = Permute((3,2,1))(model_input)
    data = Conv2D(60, (1, 4), strides=(1, 2), activation='relu',kernel_constraint=max_norm(2.))(data)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = Conv2D(40, (1, 3), activation='relu',kernel_constraint=max_norm(2.))(data)
    data = Conv2D(20, (1, 2), activation='relu',kernel_constraint=max_norm(2.))(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(100, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    return model


def ConvRNN(model_input, cfg, nb_classes=2):
    dropoutRate = 0.5
    norm_rate = 0.25
    data = Permute((3,2,1))(model_input)

    blk1 = Conv2D(8, (1, 5), padding='same', use_bias=False)(data)
    blk1 = BatchNormalization(axis=-1)(blk1)
    blk1 = DepthwiseConv2D((1, 20), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.))(blk1)
    blk1 = BatchNormalization(axis=-1)(blk1)
    blk1 = Activation('elu')(blk1)
    blk1 = AveragePooling2D((1, 4))(blk1)
    blk1 = Dropout(dropoutRate)(blk1)

    blk2 = SeparableConv2D( 16, (1, 16), use_bias=False, padding='same')(blk1)
    blk2 = BatchNormalization(axis=-1)(blk2)
    blk2 = Activation('elu')(blk2)

    print(blk2.shape)
    blk3 = Reshape((int(blk2.shape[-2]), int(blk2.shape[-1])))(blk2)

    l_lstm_sent = LSTM(32, return_sequences=True)(blk3)
    l_lstm_sent = LSTM(8, return_sequences=True)(l_lstm_sent)

    flatten = Flatten()(l_lstm_sent)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    preds = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=preds)


def EegnetOrigin(model_input, cfg, nb_classes=2, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    '''
    article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    changed as the comments
    之前EEGNet采样点默认为128
    之前channel-first时，BN的轴为1，现在channel-last，则BN的轴为默认的-1
    '''
    chans = len(cfg.elec)


    blk1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   use_bias = False)(model_input)
    blk1       = BatchNormalization()(blk1)
    blk1       = DepthwiseConv2D((chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(blk1)
    blk1       = BatchNormalization()(blk1) 
    blk1       = Activation('elu')(blk1)
    blk1       = AveragePooling2D((1, 4))(blk1)
    blk1       = Dropout(dropoutRate)(blk1)
    
    blk2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(blk1)
    blk2       = BatchNormalization()(blk2)
    blk2       = Activation('elu')(blk2)
    blk2       = AveragePooling2D((1, 8))(blk2)
    blk2       = Dropout(dropoutRate)(blk2)
        
    flatten      = Flatten(name = 'flatten')(blk2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=model_input, outputs=softmax)


def DeepconvnetOrigin(model_input,cfg, nb_classes=2, dropoutRate = 0.5):
    '''
    最原始的是用torch写的
    这里，将其torch代码转换为了keras
    除了pool层的strides由（1，3）变为了（1，2），其他的都保留了原始的设置
    '''
    chans = len(cfg.elec)

    blk1       = Conv2D(25, (1, 10))(model_input)
    blk1       = Conv2D(25, (chans, 1),use_bias=False)(blk1)
    blk1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk1)
    blk1       = Activation('elu')(blk1)
    blk1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk1)
    blk1       = Dropout(dropoutRate)(blk1)
  
    blk2       = Conv2D(50, (1, 10),use_bias=False)(blk1)
    blk2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk2)
    blk2       = Activation('elu')(blk2)
    blk2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk2)
    blk2       = Dropout(dropoutRate)(blk2)
    
    blk3       = Conv2D(100, (1, 10),use_bias=False)(blk2)
    blk3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk3)
    blk3       = Activation('elu')(blk3)
    blk3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk3)
    blk3       = Dropout(dropoutRate)(blk3)
    
    blk4       = Conv2D(200, (1, 10),use_bias=False)(blk3)
    blk4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk4)
    blk4       = Activation('elu')(blk4)
    blk4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk4)
    blk4       = Dropout(dropoutRate)(blk4)
    
    flatten      = Flatten()(blk4)
    
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=model_input, outputs=softmax)


def DeepconvnetOriginConstraint(model_input, cfg, nb_classes=2, dropoutRate=0.5):
    '''
    在DeepconvnetOrigin基础上，加了constraint以比较kernel_constraint的效果
    '''


    chans = len(cfg.elec)

    blk1       = Conv2D(25, (1, 10),kernel_constraint=max_norm(2., axis=(0, 1, 2)))(model_input)
    blk1       = Conv2D(25, (chans, 1),use_bias=False,kernel_constraint=max_norm(2., axis=(0, 1, 2)))(blk1)
    blk1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk1)
    blk1       = Activation('elu')(blk1)
    blk1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk1)
    blk1       = Dropout(dropoutRate)(blk1)
  
    blk2       = Conv2D(50, (1, 10),use_bias=False,kernel_constraint=max_norm(2., axis=(0, 1, 2)))(blk1)
    blk2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk2)
    blk2       = Activation('elu')(blk2)
    blk2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk2)
    blk2       = Dropout(dropoutRate)(blk2)
    
    blk3       = Conv2D(100, (1, 10),use_bias=False,kernel_constraint=max_norm(2., axis=(0, 1, 2)))(blk2)
    blk3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk3)
    blk3       = Activation('elu')(blk3)
    blk3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk3)
    blk3       = Dropout(dropoutRate)(blk3)
    
    blk4       = Conv2D(200, (1, 10),use_bias=False,kernel_constraint=max_norm(2., axis=(0, 1, 2)))(blk3)
    blk4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(blk4)
    blk4       = Activation('elu')(blk4)
    blk4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(blk4)
    blk4       = Dropout(dropoutRate)(blk4)
    
    flatten      = Flatten()(blk4)
    
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=model_input, outputs=softmax)



'''
最原始的是用torch写的
在EEGNet文章中变成了keras
所以直接调用EEGNet中的代码，并将参数变回去

    EEGNet中，相对于之前原始的改变：（现在保持了原始的参数）
    1.数据长度：以前250*2，现在128*2
    2.卷积核大小：以前10，现在5
    3.max_norm constraint on all convolutional layers：以前无constraint，现在有constraint
    4.pool_size：以前 1,3   现在1,2  （注意，若将Origin版本的pool size和strides降低，则1s数据可以跑，且结果较好）
    5.strides：以前 1,3  现在 1,2

    注意：此处为了400s数据可以跑，只将strides变为了（1,2）
'''

def DeepconvnetOriginPreviousPro(model_input, cfg, nb_classes=2, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    Chans = len(cfg.elec)
    block1 = Conv2D(25, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(model_input)  # it's channel first before

    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm( 2., axis=(0, 1, 3)))(block1)
    block1 = BatchNormalization( epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def DeepconvnetSmallkernal(model_input, cfg, nb_classes=2, dropoutRate = 0.5):
    """ 
    相对于之前原始的改变：
        1.数据长度：以前250*2，现在128*2
        2.卷积核大小：以前10，现在5
        3.max_norm constraint on all convolutional layers：以前无constraint，现在有constraint
        4.pool_size：以前 1,3   现在1,2  
        5.strides：以前 1,3  现在 1,2     
    
    """
    
    return Model(inputs=model_input, outputs=softmax)


def ShallowconvnetOrigin(model_input, cfg, nb_classes=2, dropoutRate = 0.5):
    """
    一下为EEGNet文章中，相对于原文的改变，
    我们此处只是数据长度不一样，保留了原文的设置

                    EEGNet文章   original paper
    数据长度          128*2       250*2
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25   
    maxnorm          有改变
    kernel_限制      constraint   无 
    
    """


    chans = len(cfg.elec)

    blk1       = Conv2D(40, (1, 25))(model_input)
    blk1       = Conv2D(40, (chans, 1), use_bias=False)(blk1)
    blk1       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(blk1)
    blk1       = Activation(square)(blk1)
    blk1       = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(blk1)
    blk1       = Activation(log)(blk1)
    blk1       = Dropout(dropoutRate)(blk1)
    flatten      = Flatten()(blk1)
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=model_input, outputs=softmax)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))


'''  #######################################  以下是3D模型的结构  ####################################### '''


def Base3D(model_input, cfg):
    blk1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    blk1 = BatchNormalization()(blk1)
    blk1 = Activation('relu')(blk1)

    blk2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(blk1)
    blk2 = BatchNormalization()(blk2)
    blk2 = Activation('relu')(blk2)

    blk3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(blk2)
    blk3 = BatchNormalization()(blk3)
    blk3 = Activation('relu')(blk3)

    flt = Flatten()(blk3)
    ds1 = Dense(32)(flt)
    ds1 = BatchNormalization()(ds1)
    ds1 = Activation('relu')(ds1)

    ds2 = Dense(32)(ds1)
    ds2 = BatchNormalization()(ds2)
    ds2 = Activation('relu')(ds2)

    out_put = Dense(2, activation='softmax')(ds2)

    return Model(model_input, out_put)


def Deep3D(model_input, cfg,conv1=16,conv2=32,conv3=64,flatten_dense=32):
    trans1 = Conv3D(conv1, (3, 3, 5), strides=(2, 2, 4),name=cfg.mdl_nm+str(1))(model_input)
    trans1 = BatchNormalization(name=cfg.mdl_nm+str(2))(trans1)
    trans1 = Activation('relu',name=cfg.mdl_nm+str(3))(trans1)

    # conv block 1-2 (cv1)
    cv1 = Conv3D(conv1, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(4))(trans1)
    cv1 = BatchNormalization(name=cfg.mdl_nm+str(5))(cv1)
    cv1 = Activation('relu',name=cfg.mdl_nm+str(6))(cv1)

    cv2 = Conv3D(conv1, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(7))(cv1)
    cv2 = BatchNormalization(name=cfg.mdl_nm+str(8))(cv2)
    cv2 = Activation('relu',name=cfg.mdl_nm+str(9))(cv2)

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(conv2, (2, 2, 3), strides=(2, 2, 2),name=cfg.mdl_nm+str(10))(cv2)
    trans2 = BatchNormalization(name=cfg.mdl_nm+str(11))(trans2)
    trans2 = Activation('relu',name=cfg.mdl_nm+str(12))(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3 = Conv3D(conv2, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(13))(trans2)
    cv3 = BatchNormalization(name=cfg.mdl_nm+str(14))(cv3)
    cv3 = Activation('relu',name=cfg.mdl_nm+str(15))(cv3)

    cv4 = Conv3D(conv2, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(16))(cv3)
    cv4 = BatchNormalization(name=cfg.mdl_nm+str(17))(cv4)
    cv4 = Activation('relu',name=cfg.mdl_nm+str(18))(cv4)

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(conv3, (2, 2, 3), strides=(2, 2, 2),name=cfg.mdl_nm+str(19))(cv4)
    trans3 = BatchNormalization(name=cfg.mdl_nm+str(20))(trans3)
    trans3 = Activation('relu',name=cfg.mdl_nm+str(21))(trans3)

    # conv block 5-6: inputsize: 4,4,99,64
    cv5 = Conv3D(conv3, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(22))(trans3)
    cv5 = BatchNormalization(name=cfg.mdl_nm+str(23))(cv5)
    cv5 = Activation('relu',name=cfg.mdl_nm+str(24))(cv5)

    cv6 = Conv3D(conv3, (2, 2, 3), padding='same',name=cfg.mdl_nm+str(25))(cv5)
    cv6 = BatchNormalization(name=cfg.mdl_nm+str(26))(cv6)
    cv6 = Activation('relu',name=cfg.mdl_nm+str(27))(cv6)

    # flatten and classification
    flt = Flatten(name=cfg.mdl_nm+str(28))(cv6)

    ds = Dense(flatten_dense,name=cfg.mdl_nm+str(29))(flt)
    ds = BatchNormalization(name=cfg.mdl_nm+str(30))(ds)
    ds = Activation('relu',name=cfg.mdl_nm+str(31))(ds)

    ds = Dense(flatten_dense,name=cfg.mdl_nm+str(32))(ds)
    ds = BatchNormalization(name=cfg.mdl_nm+str(33))(ds)
    ds = Activation('relu',name=cfg.mdl_nm+str(34))(ds)

    out_put = Dense(2, activation='softmax',name=cfg.mdl_nm+str(35))(ds)

    model = Model(model_input, out_put)
    return model


def Deep3DConstraint(model_input, cfg,conv1=32,conv2=32,conv3=64,flatten_dense=32):
    trans1 = Conv3D(conv1, (3, 3, 5), strides=(2, 2, 4),kernel_constraint=max_norm(1.0))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 (cv1)
    cv1 = Conv3D(conv1, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(trans1)
    cv1 = BatchNormalization()(cv1)
    cv1 = Activation('relu')(cv1)

    cv2 = Conv3D(conv1, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(cv1)
    cv2 = BatchNormalization()(cv2)
    cv2 = Activation('relu',)(cv2)

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(conv2, (2, 2, 3), strides=(2, 2, 2),kernel_constraint=max_norm(1.0))(cv2)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3 = Conv3D(conv2, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(trans2)
    cv3 = BatchNormalization()(cv3)
    cv3 = Activation('relu')(cv3)

    cv4 = Conv3D(conv2, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(cv3)
    cv4 = BatchNormalization()(cv4)
    cv4 = Activation('relu')(cv4)

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(conv3, (2, 2, 3), strides=(2, 2, 2),kernel_constraint=max_norm(1.0))(cv4)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64
    cv5 = Conv3D(conv3, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(trans3)
    cv5 = BatchNormalization()(cv5)
    cv5 = Activation('relu')(cv5)

    cv6 = Conv3D(conv3, (2, 2, 3), padding='same',kernel_constraint=max_norm(1.0))(cv5)
    cv6 = BatchNormalization()(cv6)
    cv6 = Activation('relu')(cv6)

    # flatten and classification
    flt = Flatten()(cv6)

    ds = Dense(flatten_dense,kernel_constraint=max_norm(1.0))(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(flatten_dense,kernel_constraint=max_norm(1.0))(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model


def Deep3DTwoBranch(model_input, cfg,conv1=32,conv2=32,conv3=64,flatten_dense=32):
    '''
    以 DeepBase3D 为基础，进行分支，每次两个分支，再融合，再分支，再融合
    '''
    trans1 = Conv3D(conv1, (3, 3, 5), strides=(2, 2, 4))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 (cv1_left) ——
    cv1_left = Conv3D(conv1, (2, 2, 3), padding='same')(trans1)
    cv1_left = BatchNormalization()(cv1_left)
    cv1_left = Activation('relu')(cv1_left)

    cv2_left = Conv3D(conv1, (2, 2, 3), padding='same')(cv1_left)
    cv2_left = BatchNormalization()(cv2_left)
    cv2_left = Activation('relu')(cv2_left)

    cv1_right = Conv3D(conv1, (3, 3, 5), padding='same')(trans1)
    cv1_right = BatchNormalization()(cv1_right)
    cv1_right = Activation('relu')(cv1_right)

    cv2_right = Conv3D(conv1, (3, 3, 5), padding='same')(cv1_right)
    cv2_right = BatchNormalization()(cv2_right)
    cv2_right = Activation('relu')(cv2_right)

    cv12_left_right = Add()([cv2_left, cv2_right])

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(conv2, (2, 2, 3), strides=(2, 2, 2))(cv12_left_right)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3_left = Conv3D(conv2, (2, 2, 3), padding='same')(trans2)
    cv3_left = BatchNormalization()(cv3_left)
    cv3_left = Activation('relu', )(cv3_left)

    cv4_left = Conv3D(conv2, (2, 2, 3), padding='same')(cv3_left)
    cv4_left = BatchNormalization()(cv4_left)
    cv4_left = Activation('relu')(cv4_left)

    cv3_right = Conv3D(conv2, (3, 3, 5), padding='same')(trans2)
    cv3_right = BatchNormalization()(cv3_right)
    cv3_right = Activation('relu')(cv3_right)

    cv4_right = Conv3D(conv2, (3, 3, 5), padding='same')(cv3_right)
    cv4_right = BatchNormalization()(cv4_right)
    cv4_right = Activation('relu')(cv4_right)

    cv34_left_right = Add()([cv4_left, cv4_right])

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(conv3, (2, 2, 3), strides=(2, 2, 2))(cv34_left_right)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64

    cv5_left = Conv3D(conv3, (2, 2, 3), padding='same')(trans3)
    cv5_left = BatchNormalization()(cv5_left)
    cv5_left = Activation('relu')(cv5_left)

    cv6_left = Conv3D(conv3, (2, 2, 3), padding='same')(cv5_left)
    cv6_left = BatchNormalization()(cv6_left)
    cv6_left = Activation('relu')(cv6_left)

    cv5_right = Conv3D(conv3, (3, 3, 5), padding='same')(trans3)
    cv5_right = BatchNormalization()(cv5_right)
    cv5_right = Activation('relu')(cv5_right)

    cv6_right = Conv3D(conv3, (3, 3, 5), padding='same')(cv5_right)
    cv6_right = BatchNormalization()(cv6_right)
    cv6_right = Activation('relu')(cv6_right)

    cv56_left_right = Add()([cv6_left, cv6_right])

    # flatten and classification
    flt = Flatten()(cv56_left_right)

    ds = Dense(flatten_dense)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(flatten_dense)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model


def Deep3DTwoBranchResnet(model_input, cfg,conv1=32,conv2=32,conv3=64,flatten_dense=32):
    '''
    以 Deep3DTwoBranch 为基础，加上resnet防止过拟合
    '''
    trans1 = Conv3D(conv1, (3, 3, 5), strides=(2, 2, 4))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 
    cv1_left = Conv3D(conv1, (2, 2, 3), padding='same')(trans1)
    cv1_left = BatchNormalization()(cv1_left)
    cv1_left = Activation('relu')(cv1_left)

    cv2_left = Conv3D(conv1, (2, 2, 3), padding='same')(cv1_left)
    cv2_left = BatchNormalization()(cv2_left)
    cv2_left = Activation('relu')(cv2_left)

    cv1_right = Conv3D(conv1, (3, 3, 5), padding='same')(trans1)
    cv1_right = BatchNormalization()(cv1_right)
    cv1_right = Activation('relu')(cv1_right)

    cv2_right = Conv3D(conv1, (3, 3, 5), padding='same')(cv1_right)
    cv2_right = BatchNormalization()(cv2_right)
    cv2_right = Activation('relu')(cv2_right)

    cv12_left_right = Add()([cv2_left, cv2_right, trans1])

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(conv2, (2, 2, 3), strides=(2, 2, 2))(cv12_left_right)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3_left = Conv3D(conv2, (2, 2, 3), padding='same')(trans2)
    cv3_left = BatchNormalization()(cv3_left)
    cv3_left = Activation('relu', )(cv3_left)

    cv4_left = Conv3D(conv2, (2, 2, 3), padding='same')(cv3_left)
    cv4_left = BatchNormalization()(cv4_left)
    cv4_left = Activation('relu')(cv4_left)

    cv3_right = Conv3D(conv2, (3, 3, 5), padding='same')(trans2)
    cv3_right = BatchNormalization()(cv3_right)
    cv3_right = Activation('relu')(cv3_right)

    cv4_right = Conv3D(conv2, (3, 3, 5), padding='same')(cv3_right)
    cv4_right = BatchNormalization()(cv4_right)
    cv4_right = Activation('relu')(cv4_right)

    cv34_left_right = Add()([cv4_left, cv4_right,trans2])

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(conv3, (2, 2, 3), strides=(2, 2, 2))(cv34_left_right)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64

    cv5_left = Conv3D(conv3, (2, 2, 3), padding='same')(trans3)
    cv5_left = BatchNormalization()(cv5_left)
    cv5_left = Activation('relu')(cv5_left)

    cv6_left = Conv3D(conv3, (2, 2, 3), padding='same')(cv5_left)
    cv6_left = BatchNormalization()(cv6_left)
    cv6_left = Activation('relu')(cv6_left)

    cv5_right = Conv3D(conv3, (3, 3, 5), padding='same')(trans3)
    cv5_right = BatchNormalization()(cv5_right)
    cv5_right = Activation('relu')(cv5_right)

    cv6_right = Conv3D(conv3, (3, 3, 5), padding='same')(cv5_right)
    cv6_right = BatchNormalization()(cv6_right)
    cv6_right = Activation('relu')(cv6_right)

    cv56_left_right = Add()([cv6_left, cv6_right,trans3])

    # flatten and classification
    flt = Flatten()(cv56_left_right)

    ds = Dense(flatten_dense)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(flatten_dense)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model


def Deep3DTwoBranchBigResnet(model_input, cfg,conv1=32,conv2=32,conv3=64,flatten_dense=32):
    '''
    以 Deep3DTwoBranchResnet 为基础，加上大分支resnet
    Deep3DTwoBranchResnet中是3个小分支
    此处，再加上了从trans1到最后的大分支
    '''
    trans1 = Conv3D(conv1, (3, 3, 5), strides=(2, 2, 4))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 
    cv1_left = Conv3D(conv1, (2, 2, 3), padding='same')(trans1)
    cv1_left = BatchNormalization()(cv1_left)
    cv1_left = Activation('relu')(cv1_left)

    cv2_left = Conv3D(conv1, (2, 2, 3), padding='same')(cv1_left)
    cv2_left = BatchNormalization()(cv2_left)
    cv2_left = Activation('relu')(cv2_left)

    cv1_right = Conv3D(conv1, (3, 3, 5), padding='same')(trans1)
    cv1_right = BatchNormalization()(cv1_right)
    cv1_right = Activation('relu')(cv1_right)

    cv2_right = Conv3D(conv1, (3, 3, 5), padding='same')(cv1_right)
    cv2_right = BatchNormalization()(cv2_right)
    cv2_right = Activation('relu')(cv2_right)

    cv12_left_right = Add()([cv2_left, cv2_right, trans1])

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(conv2, (2, 2, 3), strides=(2, 2, 2))(cv12_left_right)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3_left = Conv3D(conv2, (2, 2, 3), padding='same')(trans2)
    cv3_left = BatchNormalization()(cv3_left)
    cv3_left = Activation('relu', )(cv3_left)

    cv4_left = Conv3D(conv2, (2, 2, 3), padding='same')(cv3_left)
    cv4_left = BatchNormalization()(cv4_left)
    cv4_left = Activation('relu')(cv4_left)

    cv3_right = Conv3D(conv2, (3, 3, 5), padding='same')(trans2)
    cv3_right = BatchNormalization()(cv3_right)
    cv3_right = Activation('relu')(cv3_right)

    cv4_right = Conv3D(conv2, (3, 3, 5), padding='same')(cv3_right)
    cv4_right = BatchNormalization()(cv4_right)
    cv4_right = Activation('relu')(cv4_right)

    cv34_left_right = Add()([cv4_left, cv4_right,trans2])

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(conv3, (2, 2, 3), strides=(2, 2, 2))(cv34_left_right)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64

    cv5_left = Conv3D(conv3, (2, 2, 3), padding='same')(trans3)
    cv5_left = BatchNormalization()(cv5_left)
    cv5_left = Activation('relu')(cv5_left)

    cv6_left = Conv3D(conv3, (2, 2, 3), padding='same')(cv5_left)
    cv6_left = BatchNormalization()(cv6_left)
    cv6_left = Activation('relu')(cv6_left)

    cv5_right = Conv3D(conv3, (3, 3, 5), padding='same')(trans3)
    cv5_right = BatchNormalization()(cv5_right)
    cv5_right = Activation('relu')(cv5_right)

    cv6_right = Conv3D(conv3, (3, 3, 5), padding='same')(cv5_right)
    cv6_right = BatchNormalization()(cv6_right)
    cv6_right = Activation('relu')(cv6_right)

    BigRes = Conv3D(conv3, (2, 2, 3), padding='same')(trans1)

    cv56_left_right_BigRes = Add()([cv6_left, cv6_right,trans3,BigRes])

    # 将最后加起来的再做一个卷积
    added = Conv3D(conv2, (2, 2, 3), padding='same')(cv56_left_right_BigRes)

    # flatten and classification
    flt = Flatten()(added)

    ds = Dense(flatten_dense)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(flatten_dense)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model


def Deep3DThreeBranch(model_input, cfg):
    '''
    以 DeepBase3D 为基础，进行分支，每次两个分支，再融合，再分支，再融合
    '''
    trans1 = Conv3D(32, (3, 3, 5), strides=(2, 2, 4))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 (cv1_left) ——
    cv1_left = Conv3D(32, (2, 2, 1), padding='same')(trans1)
    cv1_left = BatchNormalization()(cv1_left)
    cv1_left = Activation('relu')(cv1_left)

    cv2_left = Conv3D(32, (2, 2, 1), padding='same')(cv1_left)
    cv2_left = BatchNormalization()(cv2_left)
    cv2_left = Activation('relu')(cv2_left)

    cv1_mid = Conv3D(32, (2, 2, 3), padding='same')(trans1)
    cv1_mid = BatchNormalization()(cv1_mid)
    cv1_mid = Activation('relu')(cv1_mid)

    cv2_mid = Conv3D(32, (2, 2, 3), padding='same')(cv1_mid)
    cv2_mid = BatchNormalization()(cv2_mid)
    cv2_mid = Activation('relu')(cv2_mid)

    cv1_right = Conv3D(32, (3, 3, 5), padding='same')(trans1)
    cv1_right = BatchNormalization()(cv1_right)
    cv1_right = Activation('relu')(cv1_right)

    cv2_right = Conv3D(32, (3, 3, 5), padding='same')(cv1_right)
    cv2_right = BatchNormalization()(cv2_right)
    cv2_right = Activation('relu')(cv2_right)

    cv12_left_right = Add()([cv2_left, cv2_mid,cv2_right])

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(cv12_left_right)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3_left = Conv3D(32, (2, 2, 1), padding='same')(trans2)
    cv3_left = BatchNormalization()(cv3_left)
    cv3_left = Activation('relu', )(cv3_left)

    cv4_left = Conv3D(32, (2, 2, 1), padding='same')(cv3_left)
    cv4_left = BatchNormalization()(cv4_left)
    cv4_left = Activation('relu')(cv4_left)

    cv3_mid = Conv3D(32, (2, 2, 3), padding='same')(trans2)
    cv3_mid = BatchNormalization()(cv3_mid)
    cv3_mid = Activation('relu', )(cv3_mid)

    cv4_mid = Conv3D(32, (2, 2, 3), padding='same')(cv3_mid)
    cv4_mid = BatchNormalization()(cv4_mid)
    cv4_mid = Activation('relu')(cv4_mid)

    cv3_right = Conv3D(32, (3, 3, 5), padding='same')(trans2)
    cv3_right = BatchNormalization()(cv3_right)
    cv3_right = Activation('relu')(cv3_right)

    cv4_right = Conv3D(32, (3, 3, 5), padding='same')(cv3_right)
    cv4_right = BatchNormalization()(cv4_right)
    cv4_right = Activation('relu')(cv4_right)

    cv34_left_right = Add()([cv4_left, cv4_mid, cv4_right])

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(cv34_left_right)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64

    cv5_left = Conv3D(64, (2, 2, 1), padding='same')(trans3)
    cv5_left = BatchNormalization()(cv5_left)
    cv5_left = Activation('relu')(cv5_left)

    cv6_left = Conv3D(64, (2, 2, 1), padding='same')(cv5_left)
    cv6_left = BatchNormalization()(cv6_left)
    cv6_left = Activation('relu')(cv6_left)

    cv5_mid = Conv3D(64, (2, 2, 3), padding='same')(trans3)
    cv5_mid = BatchNormalization()(cv5_mid)
    cv5_mid = Activation('relu')(cv5_mid)

    cv6_mid = Conv3D(64, (2, 2, 3), padding='same')(cv5_mid)
    cv6_mid = BatchNormalization()(cv6_mid)
    cv6_mid = Activation('relu')(cv6_mid)

    cv5_right = Conv3D(64, (3, 3, 5), padding='same')(trans3)
    cv5_right = BatchNormalization()(cv5_right)
    cv5_right = Activation('relu')(cv5_right)

    cv6_right = Conv3D(64, (3, 3, 5), padding='same')(cv5_right)
    cv6_right = BatchNormalization()(cv6_right)
    cv6_right = Activation('relu')(cv6_right)

    cv56_left_right = Add()([cv6_left, cv6_mid,cv6_right])

    # flatten and classification
    flt = Flatten()(cv56_left_right)

    ds = Dense(32)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(32)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model



def Deep3DSmallKernel(model_input, cfg):
    '''
    基于 Deep_Base3D 改写的，将其中的kernel m*n 都变成了 m*1 1*n 的组合，减少参数量，并且看能不能提高
    '''
    trans1 = Conv3D(32, (3, 3, 5), strides=(2, 2, 4))(model_input)
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 (cv1)
    cv1 = Conv3D(32, (1, 2, 1), padding='same')(trans1)
    cv1 = BatchNormalization()(cv1)
    cv1 = Conv3D(32, (2, 1, 1), padding='same')(cv1)
    cv1 = BatchNormalization()(cv1)
    cv1 = Conv3D(32, (1, 1, 3), padding='same')(cv1)
    cv1 = BatchNormalization()(cv1)
    cv1 = Activation('relu')(cv1)

    cv2 = Conv3D(32, (1, 2, 1), padding='same')(cv1)
    cv2 = BatchNormalization()(cv2)
    cv2 = Conv3D(32, (2, 1, 1), padding='same')(cv2)
    cv2 = BatchNormalization()(cv2)
    cv2 = Conv3D(32, (1, 1, 3), padding='same')(cv2)
    cv2 = BatchNormalization()(cv2)
    cv2 = Activation('relu')(cv2)

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(cv2)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3 = Conv3D(32, (1, 2, 1), padding='same')(trans2)
    cv3 = BatchNormalization()(cv3)
    cv3 = Conv3D(32, (2, 1, 1), padding='same')(cv3)
    cv3 = BatchNormalization()(cv3)
    cv3 = Conv3D(32, (1, 1, 3), padding='same')(cv3)
    cv3 = BatchNormalization()(cv3)
    cv3 = Activation('relu')(cv3)

    cv4 = Conv3D(32, (1, 2, 1), padding='same')(cv3)
    cv4 = BatchNormalization()(cv4)
    cv4 = Conv3D(32, (2, 1, 1), padding='same')(cv4)
    cv4 = BatchNormalization()(cv4)
    cv4 = Conv3D(32, (1, 1, 3), padding='same')(cv4)
    cv4 = BatchNormalization()(cv4)
    cv4 = Activation('relu')(cv4)

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(cv4)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64
    cv5 = Conv3D(64, (1, 2, 1), padding='same')(trans3)
    cv5 = BatchNormalization()(cv5)
    cv5 = Conv3D(64, (2, 1, 1), padding='same')(cv5)
    cv5 = BatchNormalization()(cv5)
    cv5 = Conv3D(64, (1, 1, 3), padding='same')(cv5)
    cv5 = BatchNormalization()(cv5)
    cv5 = Activation('relu')(cv5)

    cv6 = Conv3D(64, (1, 2, 1), padding='same')(cv5)
    cv6 = BatchNormalization()(cv6)
    cv6 = Conv3D(64, (2, 1, 1), padding='same')(cv6)
    cv6 = BatchNormalization()(cv6)
    cv6 = Conv3D(64, (1, 1, 3), padding='same')(cv6)
    cv6 = BatchNormalization()(cv6)
    cv6 = Activation('relu')(cv6)

    # flatten and classification
    flt = Flatten()(cv6)

    ds = Dense(32)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(32)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    out_put = Dense(2, activation='softmax')(ds)

    model = Model(model_input, out_put)
    return model



'''  #######################################  以下是2D和3D模型一起的结构  ####################################### '''

def Mix_DeepconvnetSmallkernal_Deep3D(model_input, cfg, nb_classes=2, dropoutRate = 0.5):

    chans = len(cfg.elec)

    # DeepconvnetSmallkernal 分支
    blk1       = Conv2D(25, (1, 5), kernel_constraint = max_norm(2., axis=(0,1,-1)))(model_input[0])
    blk1       = Conv2D(25, (chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,-1)))(blk1)
    blk1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blk1)
    blk1       = Activation('elu')(blk1)
    blk1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(blk1)
    blk1       = Dropout(dropoutRate)(blk1)
  
    blk2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,-1)))(blk1)
    blk2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blk2)
    blk2       = Activation('elu')(blk2)
    blk2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(blk2)
    blk2       = Dropout(dropoutRate)(blk2)
    
    blk3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,-1)))(blk2)
    blk3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blk3)
    blk3       = Activation('elu')(blk3)
    blk3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(blk3)
    blk3       = Dropout(dropoutRate)(blk3)
    
    blk4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,-1)))(blk3)
    blk4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blk4)
    blk4       = Activation('elu')(blk4)
    blk4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(blk4)
    blk4       = Dropout(dropoutRate)(blk4)
    
    flatten      = Flatten()(blk4)
    
    DeepconvnetSmallkernal_dense  = Dense(nb_classes, activation='softmax',kernel_constraint = max_norm(0.5))(flatten)

    # DeepBase3D 分支
    trans1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input[1])
    trans1 = BatchNormalization()(trans1)
    trans1 = Activation('relu')(trans1)

    # conv block 1-2 (cv1)
    cv1 = Conv3D(16, (2, 2, 3), padding='same')(trans1)
    cv1 = BatchNormalization()(cv1)
    cv1 = Activation('relu')(cv1)

    cv2 = Conv3D(16, (2, 2, 3), padding='same')(cv1)
    cv2 = BatchNormalization()(cv2)
    cv2 = Activation('relu')(cv2)

    # transition bolock 1 : inputsize: 4,4,99,64
    trans2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(cv2)
    trans2 = BatchNormalization()(trans2)
    trans2 = Activation('relu')(trans2)

    # conv block 3-4: inputsize: 4,4,99,64
    cv3 = Conv3D(32, (2, 2, 3), padding='same')(trans2)
    cv3 = BatchNormalization()(cv3)
    cv3 = Activation('relu')(cv3)

    cv4 = Conv3D(32, (2, 2, 3), padding='same')(cv3)
    cv4 = BatchNormalization()(cv4)
    cv4 = Activation('relu')(cv4)

    # transition bolock 3 : inputsize: 4,4,99,64
    trans3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(cv4)
    trans3 = BatchNormalization()(trans3)
    trans3 = Activation('relu')(trans3)

    # conv block 5-6: inputsize: 4,4,99,64
    cv5 = Conv3D(64, (2, 2, 3), padding='same')(trans3)
    cv5 = BatchNormalization()(cv5)
    cv5 = Activation('relu')(cv5)

    cv6 = Conv3D(64, (2, 2, 3), padding='same')(cv5)
    cv6 = BatchNormalization()(cv6)
    cv6 = Activation('relu')(cv6)

    # flatten and classification
    flt = Flatten()(cv6)

    ds = Dense(32)(flt)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    ds = Dense(32)(ds)
    ds = BatchNormalization()(ds)
    ds = Activation('relu')(ds)

    DeepBase3D_dense = Dense(2, activation='softmax')(ds)

    # 两个分支的连接
    output = Concatenate(axis=-1)([DeepconvnetSmallkernal_dense, DeepBase3D_dense])
    output = Dense(2,activation='softmax')(output)

    return Model(model_input,output)