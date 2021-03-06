from logging import getLogger

import numpy as np
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.optimizers import Adam

from keras_glow.config import Config
from keras_glow.data.data_processor import DataProcessor
from keras_glow.glow.agent import Agent
from keras_glow.glow.model import GlowModel
from keras_glow.glow.model_parts import Invertible1x1Conv, GaussianDiag

logger = getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def fit(self, model: GlowModel, dp: DataProcessor):
        tc = self.config.training
        model.dump_model_internal()
        self.compile(model)
        steps_per_epoch = tc.steps_per_epoch or dp.image_count//tc.batch_size

        callbacks = [
            SamplingCallback(self.config, model),
            TensorBoard(str(self.config.resource.tensorboard_dir), batch_size=tc.batch_size, write_graph=True,
                        # histogram_freq=5, write_grads=True
                        ),
            ReduceLROnPlateau(monitor='loss', factor=tc.lr_decay, patience=tc.lr_patience, verbose=1,
                              min_lr=tc.lr_patience),
        ]
        try:
            model.encoder.fit_generator(self.generator_for_fit(dp), epochs=tc.epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        callbacks=callbacks, verbose=1)
        except InvalidArgumentError as e:
            model.dump_model_internal()
            raise e

    def data_dependent_init(self, model: GlowModel, dp: DataProcessor):
        logger.info('Start Data Dependent Init')
        model.dump_model_internal()
        self.compile(model, lr=0.000000001)

        try:
            model.encoder.fit_generator(self.generator_for_fit(dp), epochs=1,
                                        steps_per_epoch=1, verbose=1)
        except InvalidArgumentError as e:
            model.dump_model_internal()
            raise e
        logger.info('Finish Data Dependent Init')

    def generator_for_fit(self, dp):
        while True:
            for img in dp.iterator(batch_size=self.config.training.batch_size):
                # logger.debug(f'img.shape={img.shape}, img.mean={np.mean(img)}')
                yield (img, np.zeros((img.shape[0],)))

    def compile(self, model: GlowModel, lr=None):
        tc = self.config.training
        lr = lr or tc.lr
        model.encoder.compile(optimizer=Adam(lr=lr), loss=create_prior_loss(model.bit_per_sub_pixel_factor))


class SamplingCallback(Callback):
    def __init__(self, config: Config, glow_model: GlowModel):
        self.config = config
        self.glow_model = glow_model
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # logger.debug(f"logs={logs}")
        sample_n_epoch = self.config.training.sample_every_n_epoch
        self.save_model()
        if not sample_n_epoch:
            return
        if epoch % sample_n_epoch == 0:
            self.sample_image(epoch)
        if logs and 'loss' in logs:
            logs['loss'] = np.mean(logs['loss'])

    def sample_image(self, epoch):
        agent = Agent(self.config, self.glow_model)
        image_path_base = f'{self.config.resource.sample_dir}/ep{epoch:04d}/img'
        agent.sample_to_save(self.config.training.sample_n_image, image_path_base=image_path_base)

    def save_model(self):
        print("")
        self.glow_model.save_all()
        print("")

    def on_batch_end(self, batch, logs=None):
        if batch % 1 == 0:
            self.glow_model.dump_model_internal()


def create_prior_loss(bit_per_sub_pixel_factor):
    def prior_loss(y_true, y_pred):
        prior = GaussianDiag.prior(K.shape(y_pred))
        return -prior.logp(y_pred) * bit_per_sub_pixel_factor
    return prior_loss
