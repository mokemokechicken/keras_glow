from logging import getLogger

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import Adam

from keras_glow.config import Config
from keras_glow.data.data_processor import DataProcessor
from keras_glow.glow.model import GlowModel


logger = getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def fit(self, model: GlowModel, dp: DataProcessor):
        tc = self.config.training
        self.compile(model)

        def generator_for_fit():
            while True:
                for img in dp.iterator(batch_size=tc.batch_size):
                    logger.debug(f'img.shape={img.shape}, img.mean={np.mean(img)}')
                    yield (img, np.zeros((img.shape[0], )))

        callbacks = [
            TensorBoard(str(self.config.resource.tensorboard_dir), batch_size=tc.batch_size, write_graph=True),
        ]
        model.encoder.fit_generator(generator_for_fit(), epochs=tc.epochs,
                                    steps_per_epoch=1,  # dp.image_count//tc.batch_size,
                                    callbacks=callbacks, verbose=1)

    def compile(self, model: GlowModel):
        tc = self.config.training
        model.encoder.compile(optimizer=Adam(lr=tc.lr), loss=zero_loss)


def zero_loss(y_true, y_pred):
    #return K.constant(0., dtype='float32')
    return mean_squared_error(y_true, y_pred)



