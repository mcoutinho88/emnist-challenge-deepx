from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from google_drive_downloader import GoogleDriveDownloader as gdd

from ..blocks.common_blocks import conv_block, dense_block

class ConvNet():
    ''' Convolutional Network model
            Constructor:
                input-shape: shape of the input to the convolutional layer /
                n_class: number of classes/
            Returns:
                Model of the ConvNet to be trained
    '''
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

    def create_model(self):
        x = layers.Input(shape=self.input_shape)

        block1 = conv_block(x, 64, 2, drop=True)
        block2 = conv_block(block1, 128, 2, drop=True)
        block3 = dense_block(block2, self.n_class)

        output_layer = layers.Activation('sigmoid')(block3)

        train_model = Model(x, output_layer)

        return train_model

    def download_model(self, file_id, model_path, download_model_from_drive=False):
        ''' Load and/or download model that already exists and is trained
            Arguments:
                file_id: id from the file hosted on Google Drive
                model_path: path to the model to be used
                download_model_from_drive: set this True if you want to download the model from the Google Drive
            Returns:
                Trained model
        '''
        if (download_model_from_drive):
            # download saved model from my Google Drive
            gdd.download_file_from_google_drive(file_id=file_id,
                                                dest_path=model_path,
                                                unzip=False)

        loaded_model = load_model(model_path)

        return loaded_model
