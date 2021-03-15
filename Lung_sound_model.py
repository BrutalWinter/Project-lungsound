import tensorflow as tf
#################################################

class Conv2D_loopbody(tf.keras.layers.Layer):
    # DarknetConv1D_BN_Leaky(32, 5, strides=(1,1))
    #       DarknetConv1D(*args, **self.no_bias_kwargs)
    def __init__(self, *args, name="Conv2D_loopbody", **kwargs):
        super(Conv2D_loopbody, self).__init__(name=name) ### initialize its parents
        self.kwargs = {}
        self.kwargs['use_bias'] = 'False'
        self.kwargs['kernel_initializer'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # self.kwargs['padding'] = 'valid' if kwargs.get('strides') == 2 else 'same'
        self.kwargs['padding'] = 'same'
        self.kwargs.update(kwargs)


        self.conv=tf.keras.layers.Conv2D(*args, **self.kwargs)
        self.Relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.Pool = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=2, padding='valid', data_format=None, **kwargs)
        self.layer_Residual = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.Relu(x)
        x = self.Pool(x)
        return self.layer_Residual([inputs, x])




class FC_Relu_layers(tf.keras.layers.Layer):
    def __init__(self, num_filters, name="FC_Relu_layers"):
        super(FC_Relu_layers, self).__init__(name=name)

        self.FC = tf.keras.layers.Dense(num_filters, activation=None)
        self.Relu = tf.keras.layers.LeakyReLU(alpha=0.1)


    def call(self, inputs):
        x = self.self.FC(inputs)
        x = self.Relu(x)

        return x




class LungSound_Model(tf.keras.Model):
    def __init__(self, outputs, kernelsize, name="Main_model"):
        super(LungSound_Model, self).__init__(name=name)

        self.layer0 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer1 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer2 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer3 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer4 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer5 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer6 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer7 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer8 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer9 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer10 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer11 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))
        self.layer12 =Conv2D_loopbody(outputs, kernelsize, strides=(1,1))

        self.Fc_Relu0 =FC_Relu_layers(num_filters=1024)
        self.Fc_Relu1 =FC_Relu_layers(num_filters=512)
        self.Fc_Relu2 =FC_Relu_layers(num_filters=256)

        self.Fc_end=tf.keras.layers.Dense(4, activation=None)
        self.Soft = tf.keras.layers.Softmax(axis=-1)




    def call(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)

        x = self.Fc_Relu0(x)
        x = self.Fc_Relu1(x)
        x = self.Fc_Relu2(x)

        x = self.Fc_end(x)
        x = self.Soft(x)

        return x



if __name__ == '__main__':
    LungSound_model= LungSound_Model(32,5)
    ##############################################################
    data_set=tf.random.uniform(shape=[10,224,140], minval=0, maxval=10, dtype=tf.int32)
    batch_size = 2
    data_set = data_set.batch(batch_size)


    for epoch in tf.range(10, 20):
        for step, data in enumerate(data_set):
            PCM_batch= data
            print(PCM_batch)




