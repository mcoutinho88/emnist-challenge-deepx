from tensorflow.keras import layers


def conv_block(layer_in, n_filters, n_conv, drop=False):
    ''' Contains a number of convolutional layers and a Max-Pooling 2x2
        Arguments:
            layer_in: input layer of the convolutional layer
            n_filters: number of filters to be used in the convolution
            n_filters: number of sequential convolutions
            drop: set this True if you want to use Dropout
        Returns:
            layer_out: output layer after all convolutions
    '''

    # add convolutional layers
    for _ in range(n_conv):
        layer_in = layers.Conv2D(n_filters, (3, 3), padding='same')(layer_in)
        layer_in = layers.Activation("relu")(layer_in)
    # add max pooling layer
    layer_out = layers.MaxPooling2D((2, 2), strides=(2, 2))(layer_in)

    if (drop):
        layer_out = layers.Dropout(0.2)(layer_out)
    return layer_out



def dense_block(layer_in, n_classes):

    layer_in = layers.Flatten()(layer_in)
    layer_out = layers.Dense(n_classes)(layer_in)
    return layer_out