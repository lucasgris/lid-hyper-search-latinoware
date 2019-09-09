"""
Basic module for configuration properties.

Create your configuration object with Config()
The configuration object have some default properties.
"""
import os
import pickle
from datetime import datetime

"""
Default configuration properties:
"""
DATASET_NAME = 'CPD_v0-1a'
SAMPLING_RATE = 8000
MAX_SECONDS_PER_RUN = 259200
NUM_CLASSES = 3
STEPS_PER_EPOCH = None
SECONDS = 5
EPOCHS = 8
BATCH_SIZE = 1  # Will not use
LEARNING_RATE = 0.0001
LOG_DIR = 'logs'
MODELS_DIR = 'models'
MODELS_CHECKPOINT_DIR = 'models/checkpoints'
PREFIX_PATH = os.getcwd()
DATA_PATH = os.path.join('.', 'dev')
TEST_DATA_CSV = os.path.join(DATA_PATH,
                             f'TE-{DATASET_NAME}_shuffled_balanced.csv')
TRAIN_DATA_CSV = os.path.join(DATA_PATH,
                              f'TR-{DATASET_NAME}_shuffled_balanced.csv')
EVAL_DATA_CSV = None
USE_GENERATOR = True
USE_HEAP = False
TB_EMBEDDINGS = False

CHANNELS = 1
if SAMPLING_RATE == 8000:
    SPEC_SHAPE_HEIGTH = 81
    SPEC_SHAPE_WIDTH = 499

if SAMPLING_RATE == 16000:
    SPEC_SHAPE_HEIGTH = 161
    SPEC_SHAPE_WIDTH = 499

if SAMPLING_RATE == 48000:
    SPEC_SHAPE_HEIGTH = 481
    SPEC_SHAPE_WIDTH = 499

LOAD_CONF = False
CONF_LOCATION = None
TRAIN = True
SAVE_MODEL = True
EVALUATE = True
EVALUATE_BEST_MODEL = True
HEAP_DIR = 'tmp'
TIME_NOW = str(datetime.now()).replace(':', '-').replace(' ', '.')
LOG_FILE = f'search_log.log'
REPORT_FILE = f'report_{TIME_NOW}.csv'


"""
Config class
"""


class Config:
    """
        Config class for settings management
    """

    def __init__(self,
                 data_loader,
                 params,
                 conf_name='conf',
                 report_file=REPORT_FILE,
                 use_heap=USE_HEAP,
                 use_generator=USE_GENERATOR,
                 use_tb_embeddings=TB_EMBEDDINGS,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE,
                 num_classes=NUM_CLASSES):
        """
        This creates a new object of Config with basic properties.
        To change default properties, use the set methods.

        :param data_loader: function
            Data loader callable object.
        :param params: dict
            Hyperparameters dict
        :param conf_name: name of this configuration object. The configuration
            will be automatically stored with this name.
        :param use_heap: bool
            Choose to use or not heap functionality.
        :param use_generator: bool
            Choose to use or not the generator functionality.
        :param use_tb_embeddings: bool
            Choose to use or not embeddings in tensorboard callback.
        :param epochs: int
            Number of epochs to train.
        :param batch_size: int
            Batch size to train.
        :param learning_rate: float
            Learning rate to train.
        :param num_classes: int
            Number of class of the model - Softmax output.
        """
        # Default configurations
        self.data_loader = data_loader
        self.params = params
        self.batch_size = batch_size
        self.use_generator = use_generator
        self.use_heap = use_heap
        self.report_file = report_file
        self.use_tb_embeddings = use_tb_embeddings
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self._time_limit = None
        self._model_name = None
        self._run = 0
        self._heap_dir = os.path.join(PREFIX_PATH, HEAP_DIR)
        self._log_dir = os.path.join(PREFIX_PATH, LOG_DIR)
        self._models_dir = os.path.join(PREFIX_PATH,
                                        MODELS_DIR)
        self._models_checkpoint_dir = os.path.join(PREFIX_PATH,
                                                   MODELS_CHECKPOINT_DIR)
        self._data_path = os.path.join(PREFIX_PATH, DATA_PATH)
        self._max_seconds_per_run = MAX_SECONDS_PER_RUN
        self._steps_per_epoch = STEPS_PER_EPOCH
        self._dataset_name = DATASET_NAME
        self._train_data_csv = TRAIN_DATA_CSV
        self._eval_data_csv = EVAL_DATA_CSV
        self._test_data_csv = TEST_DATA_CSV

        self._when = datetime.now()

        self._name = conf_name
        self.dump()

    @property
    def model_name(self):
        if self._model_name is None:
            return ''
        return self._model_name

    @property
    def data_loader(self):
        return self._data_loader

    @property
    def model_location(self):
        if self._models_dir and self.model_name:
            return os.path.join(self._models_dir, self.model_name)
        return None

    @property
    def model_checkpoint_location(self):
        if self._models_checkpoint_dir and self.model_name:
            return os.path.join(self._models_checkpoint_dir, self.model_name)
        return None

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def use_generator(self):
        return self._use_generator

    @property
    def use_heap(self):
        return self._use_heap

    @property
    def use_tb_embeddings(self):
        return self._use_tb_embeddings

    @property
    def epochs(self):
        return self._epochs

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data_path(self):
        return self._data_path

    @property
    def max_seconds_per_run(self):
        return self._max_seconds_per_run

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    @property
    def dataset_name(self):
        return str(self._dataset_name)

    @property
    def train_data_csv(self):
        return self._train_data_csv

    @property
    def eval_data_csv(self):
        return self._eval_data_csv

    @property
    def test_data_csv(self):
        return self._test_data_csv

    @property
    def models_dir(self):
        return self._models_dir

    @property
    def heap_dir(self):
        return self._heap_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def run(self):
        return self._run

    @run.setter
    def run(self, number):
        self._run = number

    @property
    def models_checkpoint_dir(self):
        return self._models_checkpoint_dir

    @max_seconds_per_run.setter
    def max_seconds_per_run(self, max_seconds_per_run: int):
        self._max_seconds_per_run = max_seconds_per_run

    @property
    def time_limit(self):
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit: datetime):
        self._time_limit = time_limit

    @dataset_name.setter
    def dataset_name(self, dataset_name: str):
        self._dataset_name = dataset_name

    @num_classes.setter
    def num_classes(self, num_classes: int):
        self._num_classes = num_classes

    @models_dir.setter
    def models_dir(self, models_dir: str):
        if not os.path.isdir(models_dir):
            raise ValueError(f'Could not open {models_dir}: '
                             'Check if the directory exists')
        self._models_dir = models_dir

    @heap_dir.setter
    def heap_dir(self, heap_dir: str):
        if not os.path.isdir(heap_dir):
            raise ValueError(f'Could not open {heap_dir}: '
                             'Check if the directory exists')
        self._heap_dir = heap_dir

    @models_checkpoint_dir.setter
    def models_checkpoint_dir(self, models_checkpoint_dir: str):
        if not os.path.isdir(models_checkpoint_dir):
            raise ValueError(f'Could not open {models_checkpoint_dir}: '
                             'Check if the directory exists')
        self._models_checkpoint_dir = models_checkpoint_dir

    @log_dir.setter
    def log_dir(self, log_dir: str):
        if not os.path.isdir(log_dir):
            raise ValueError(f'Could not open {log_dir}: '
                             'Check if the directory exists')
        self._log_dir = log_dir

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = str(model_name)

    @data_loader.setter
    def data_loader(self, data_loader: callable):
        if not callable(data_loader):
            raise ValueError
        self._data_loader = data_loader

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = int(batch_size)

    @use_tb_embeddings.setter
    def use_tb_embeddings(self, use_tb_embeddings: bool):
        self._use_tb_embeddings = True if use_tb_embeddings else False

    @use_heap.setter
    def use_heap(self, use_heap: bool):
        self._use_heap = True if use_heap else False

    @use_generator.setter
    def use_generator(self, use_generator: bool):
        self._use_generator = True if use_generator else False

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = float(learning_rate)

    @data_path.setter
    def data_path(self, data_path: str):
        if not os.path.isdir(data_path):
            raise ValueError(f'Could not open {data_path}: '
                             'Check if the directory exists')
        self._data_path = data_path

    @steps_per_epoch.setter
    def steps_per_epoch(self, steps_per_epoch: int):
        self._steps_per_epoch = int(steps_per_epoch)

    @epochs.setter
    def epochs(self, epochs: int):
        self._epochs = int(epochs)

    @train_data_csv.setter
    def train_data_csv(self, train_data_csv: str):
        try:
            open(train_data_csv, 'r')
            self._train_data_csv = train_data_csv
        except OSError as err:
            raise ValueError(f'Could not open {train_data_csv}: {str(err)}')

    @test_data_csv.setter
    def test_data_csv(self, test_data_csv: str):
        try:
            open(test_data_csv, 'r')
            self._test_data_csv = test_data_csv
        except OSError as err:
            raise ValueError(f'Could not open {test_data_csv}: {str(err)}')

    @eval_data_csv.setter
    def eval_data_csv(self, eval_data_csv: str):
        try:
            open(eval_data_csv, 'r')
            self._eval_data_csv = eval_data_csv
        except OSError as err:
            raise ValueError(f'Could not open {eval_data_csv}: {str(err)}')

    @classmethod
    def frompicke(cls, file_path: str):
        with open(file_path, mode='rb') as input_file:
            obj = pickle.load(input_file)
        return obj

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def __str__(self):
        return (f'{self.name}.d-{self.dataset_name}.'
                f'{self._when.day}-{self._when.month}-'
                f'{self._when.hour}-{self._when.minute}-'
                f'{self._when.second}')

    def __repr__(self):
        return (f'Attributes of [{str(self)}] configuration:\n'
                '\tname={}\n'
                '\tdata_loader={}\n'
                '\tmodel_location={}\n'
                '\tbatch_size={}\n'
                '\tuse_generator={}\n'
                '\tuse_heap={}\n'
                '\tuse_tb_embeddings={}\n'
                '\tlog_dir={}\n'
                '\tlearning_rate={}\n'
                '\tnum_classes={}\n'
                '\tmodels_dir={}\n'
                '\tmodels_checkpoint_dir={}\n'
                '\tdata_path={}\n'
                '\tsteps_per_epoch={}\n'
                '\tepochs={}\n'
                '\ttrain_data_csv={}\n'
                '\teval_data_csv={}\n'
                '\ttest_data_csv={}\n'
                '\twhen={}\n'
                'Parameters:\n{} [...]\n\n'
                'Attributes with None value has not been set yet.'
                f'\n{("-"*80)}\n'
                .format(
                    self.name,
                    self.data_loader,
                    self.model_location,
                    self.batch_size,
                    self.use_generator,
                    self.use_heap,
                    self.use_tb_embeddings,
                    self.log_dir,
                    self.learning_rate,
                    self.num_classes,
                    self.models_dir,
                    self.models_checkpoint_dir,
                    self.data_path,
                    self.steps_per_epoch,
                    self.epochs,
                    self.train_data_csv,
                    self.eval_data_csv,
                    self.test_data_csv,
                    self._when,
                    str(self.params)[:255]))

    def dump(self):
        os.makedirs('confs', exist_ok=True)
        pickle.dump(self, open(os.path.join('confs', f'{str(self)}.pkl'),
                    'wb'), protocol=3)
    
    def delete_file(self):
        os.remove(os.path.join('confs', f'{str(self)}.pkl'))
