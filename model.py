from keras.models import Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate

class UNet():
    
    def __init__(self, loss, metrics, lr=1e-4):
        self.lr = lr
        self.loss = loss
        self.metrics = metrics

    def VGG16_encoder(self, input_shape):
        vgg16 = VGG16(weights ='imagenet',
                          include_top=False,
                          input_shape=input_shape)
    
        last = 'input_'
        encoder_layers = {last: vgg16.input}
        for layer in vgg16.layers:
            if layer.name.find('input') == -1:
                encoder_layers[layer.name] = vgg16.get_layer(layer.name)(encoder_layers[last])
                encoder_layers[layer.name].trainable = False
                last = layer.name
        
        return vgg16, encoder_layers
    
    
    def decoder(self, input_shape):
        vgg16, encoder_layers = self.VGG16_encoder(input_shape)
        
        conv0 = Conv2D(512, (2, 2), strides=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(vgg16.get_output_at(0))
        convt0 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (conv0)
        
        # 16x16x512 -> 32x32x256
        convt1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (convt0)
        merge1 = concatenate([convt1, encoder_layers['block5_conv3']])
        conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (merge1)
        conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (conv1)
        
        # 32x32x512 -> 64x64x512
        convt2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (conv1)
        merge2 = concatenate([convt2,  encoder_layers['block4_conv3']])
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (merge2)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (conv2)
        
        # 64x64x512 -> 128x128x256
        convt3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv2)
        merge3 = concatenate([convt3,  encoder_layers['block3_conv3']])
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (merge3)
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (conv3)
        
        # 128x128x256 -> 256x256x128
        convt4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv3)
        merge4 = concatenate([convt4, encoder_layers['block2_conv2']], axis=3)
        conv4 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (merge4)
        conv4 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (conv4)
        
        # 256x256x128 -> 512x512x64
        convt5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv4)
        merge5 = concatenate([convt5, encoder_layers['block1_conv2']], axis=3)
        conv5 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (merge5)
        conv5 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal') (conv5)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv5)
        
        model = Model(inputs=[vgg16.input], outputs=[outputs])
        model.compile(optimizer = Adam(lr = self.lr), loss=self.loss, metrics=self.metrics)
        
        return model
    
    def get_model(self, input_shape):
        return self.decoder(input_shape)