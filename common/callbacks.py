from common.util import logm
from common.config import MAX_SECONDS_PER_RUN
from inspect import currentframe, getframeinfo
import time
from keras.callbacks import Callback


class TimedStopping(Callback):
    """
    Timed stop callback. Stops the training processs after some time.
    Use this for performance experiments and analysis.
    """
    # https://github.com/keras-team/keras/issues/1625
    def __init__(self, seconds=MAX_SECONDS_PER_RUN,
                 safety_factor=1, verbose=1):
        """
        Creates a new TimedStopping with the provided time.

        Args:
            Callback ([type]): [description]
            seconds (int, optional): Max time to run the trainning. Defaults to
                MAX_SECONDS_PER_RUN.
            safety_factor (int, optional): The safety factor guarantees that
                the trainning process stops after the provided time plus
                the average elapsed time times this factor. This will make the
                trainning process stops earlier by a factor of the average
                duration per epoch. Defaults to 1.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(Callback, self).__init__()

        self.start_time = 0
        self.safety_factor = safety_factor
        self.seconds = seconds
        self.verbose = verbose
        self.time_logs = []

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.start_time
        self.time_logs.append(elapsed_time)

        avg_elapsed_time = (float(sum(self.time_logs)) /
                            max(len(self.time_logs), 1))

        logm('Average elapsed time: '
             f'{str(self.seconds - self.safety_factor * avg_elapsed_time)}',
             cur_frame=currentframe(), mtype='I')
        if elapsed_time > self.seconds - self.safety_factor * avg_elapsed_time:
            self.model.stop_training = True
            if self.verbose:
                logm('Stopping after %s seconds.' % self.seconds,
                     cur_frame=currentframe(), mtype='I')
