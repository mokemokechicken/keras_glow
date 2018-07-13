from logging import getLogger

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam

from keras_glow.config import Config
from keras_glow.data.data_processor import DataProcessor
from keras_glow.glow.agent import Agent
from keras_glow.glow.model import GlowModel

logger = getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def fit(self, model: GlowModel, dp: DataProcessor):
        tc = self.config.training
        self.compile(model)
        steps_per_epoch = tc.steps_per_epoch or dp.image_count//tc.batch_size

        def generator_for_fit():
            while True:
                for img in dp.iterator(batch_size=tc.batch_size):
                    # logger.debug(f'img.shape={img.shape}, img.mean={np.mean(img)}')
                    yield (img, np.zeros((img.shape[0], )))

        callbacks = [
            SamplingCallback(self.config, model),
            TensorBoard(str(self.config.resource.tensorboard_dir), batch_size=tc.batch_size, write_graph=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, verbose=1),
        ]
        model.encoder.fit_generator(generator_for_fit(), epochs=tc.epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    callbacks=callbacks, verbose=1)

    def compile(self, model: GlowModel):
        tc = self.config.training
        model.encoder.compile(optimizer=Adam(lr=tc.lr), loss=zero_loss)


class SamplingCallback(Callback):
    def __init__(self, config: Config, glow_model: GlowModel):
        self.config = config
        self.glow_model = glow_model
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logger.debug(f"logs={logs}")
        sample_n_epoch = self.config.training.sample_every_n_epoch
        self.save_model()
        if not sample_n_epoch:
            return
        if epoch % sample_n_epoch == 0:
            self.sample_image(epoch)
        if logs and 'loss' in logs:
            logs['loss'] = np.sum(logs['loss'])

    def sample_image(self, epoch):
        agent = Agent(self.config, self.glow_model)
        image_path_base = f'{self.config.resource.sample_dir}/ep{epoch:04d}/img'
        agent.sample_to_save(self.config.training.sample_n_image, image_path_base=image_path_base)

    def save_model(self):
        self.glow_model.save_all()


def zero_loss(y_true, y_pred):
    return K.constant(0., dtype='float32')
