from inspect import currentframe, getframeinfo
from datetime import datetime
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model, save_model
import os
import random
import numpy as np

from common.config import *
from common.util import logm, parse_csv, Timer
from common.dataloader import wav_to_specdata, normalize, spec_load_and_rshp,\
                              spec_save
from common.datasets import Dataset, TestDataset
from common.generator import Generator, Heap
from common.callbacks import TimedStopping
from evaluation import evaluate
from model import build_model


DEVELOPING = False


def setup_dirs(conf: Config):
    logm('Setting up directories...', cur_frame=currentframe(),
         mtype='I')

    models_checkpoint_dir = os.path.join(MODELS_CHECKPOINT_DIR,
                                         str(conf))
    os.makedirs(models_checkpoint_dir, exist_ok=True)
    conf.models_checkpoint_dir = models_checkpoint_dir
    logm('Setting up directories: the model checkpoints directory is '
         f'{conf.models_checkpoint_dir}',
         cur_frame=currentframe(), mtype='I')
    log_dir = os.path.join(LOG_DIR, str(conf))
    os.makedirs(log_dir, exist_ok=True)
    conf.log_dir = log_dir
    logm(f'Setting up directories: the log dir is {conf.log_dir}',
         cur_frame=currentframe(), mtype='I')

    models_dir = os.path.join(MODELS_DIR, str(conf))
    os.makedirs(models_dir, exist_ok=True)
    conf.models_dir = models_dir
    logm('Setting up directories: the model directory is '
         f'{conf.models_dir}',
         cur_frame=currentframe(), mtype='I')

    heap_dir = HEAP_DIR
    os.makedirs(heap_dir, exist_ok=True)
    conf.heap_dir = heap_dir
    logm('Setting up directories: the heap directory is '
         f'{conf.heap_dir}',
         cur_frame=currentframe(), mtype='I')

    logm('Setting up directories... Done', cur_frame=currentframe(),
         mtype='I')


def setup_clbks(conf: Config,
                setup_tb=False,
                setup_mc=True,
                setup_ts=False,
                setup_es=False):
    logm('Setting up callbacks', cur_frame=currentframe(), mtype='I')
    callbacks = []
    if setup_tb:
        logm('Setting up tensorboard', cur_frame=currentframe(), mtype='I')
        if conf.use_tb_embeddings:
            paths, labels = parse_csv(conf.test_data_csv,
                                      conf.data_path)
            test_set = TestDataset(paths, labels)

            with open(os.path.join(conf.log_dir, 'metadata.tsv'), 'w') as f:
                np.savetxt(f, labels, delimiter=",", fmt='%s')

            logm(f'Loading test data ({len(paths)} samples) '
                 'for tensorboard callback...',
                 cur_frame=currentframe(), mtype='I')
            y_test = test_set()[1]
            x_test = np.asarray([conf.data_loader(x) for x in test_set()[0]])

            logm(f'Loading test data ({len(paths)} samples) for '
                 'tensorboard callback... Done',
                 cur_frame=currentframe(), mtype='I')

            print('x_test shape:', x_test.shape)
            tb = TensorBoard(
                log_dir=os.path.join(conf.log_dir, 'tensorboard'),
                histogram_freq=1,
                batch_size=conf.batch_size,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=5,
                embeddings_layer_names=['features'],
                embeddings_metadata='metadata.tsv',
                embeddings_data=x_test)
        else:
            tb = TensorBoard(log_dir=conf.log_dir,
                             histogram_freq=1,
                             batch_size=conf.batch_size,
                             write_graph=True)
        callbacks.append(tb)
    if setup_ts:
        logm('Setting up TimedStopping',
             cur_frame=currentframe(),
             mtype='I')
        if conf.max_seconds_per_run:
            callbacks.append(TimedStopping(conf.max_seconds_per_run))
        else:
            logm('Could not set up TimedStopping: '
                 'conf.max_seconds_per_run is set as None',
                 cur_frame=currentframe(),
                 mtype='W')
    if setup_es:
        logm('Setting up EarlyStopping',
             cur_frame=currentframe(),
             mtype='I')
        callbacks.append(EarlyStopping(patience=5))
    if setup_mc:
        logm('Setting up ModelCheckpoint',
             cur_frame=currentframe(),
             mtype='I')
        callbacks.append(ModelCheckpoint(
            f'{conf.model_checkpoint_location}.h5',
            period=1,
            save_best_only=True))

    logm('Setting up callbacks... Done',
         cur_frame=currentframe(),
         mtype='I')
    return callbacks


def setup_heap(conf: Config):
    logm('Setting up heap...', cur_frame=currentframe(), mtype='I')
    logm(f'Setting up heap: the heap directory is {conf.heap_dir}',
         cur_frame=currentframe(), mtype='I')
    if not conf.use_heap:
        logm('Setting up heap: heap is deactivated',
             cur_frame=currentframe(), mtype='W')
        return None
    heap = Heap(load_fn=spec_load_and_rshp,
                save_fn=spec_save,
                file_format='png',
                heap_dir=HEAP_DIR)
    logm('Setting up heap... Done', cur_frame=currentframe(), mtype='I')
    return heap


def train(model: keras.Model, conf: Config, developing=False):
    logm(f'Running train for {conf}', cur_frame=currentframe(),
         mtype='I')

    paths, labels = parse_csv(conf.train_data_csv, conf.data_path)
    if developing:
        logm('Developing is set as true: limiting size of dataset',
             cur_frame=currentframe(), mtype='I')
        paths_labels = list(zip(paths, labels))
        random.shuffle(paths_labels)
        paths, labels = zip(*paths_labels)
        paths = paths[:100]
        labels = labels[:100]
        epochs = 2
        conf.steps_per_epoch = 10

    dataset = Dataset(paths, labels, shuffle=True, val_split=0.005,
                      name=conf.dataset_name)

    logm(f'Loading validation data...', cur_frame=currentframe(),
         mtype='I')
    X_val = np.asarray([conf.data_loader(x) for x in dataset.validation[0]])
    y_val = dataset.validation[1]
    logm(f'Loading validation data... Done', cur_frame=currentframe(),
         mtype='I')
    logm(f'Validation data shape is {X_val.shape}',
         cur_frame=currentframe(),
         mtype='I')

    if conf.use_generator:
        logm('Using generator', cur_frame=currentframe(), mtype='I')
        batch_size = (conf.batch_size if conf.batch_size <
                      len(dataset.train[0]) else len(dataset.train[0]))
        train_gen = Generator(paths=dataset.train[0],
                              labels=dataset.train[1],
                              loader_fn=conf.data_loader,
                              batch_size=batch_size,
                              heap=setup_heap(conf),
                              expected_shape=(SPEC_SHAPE_HEIGTH,
                                              SPEC_SHAPE_WIDTH,
                                              3))
        history = model.fit_generator(generator=train_gen,
                                      validation_data=(X_val,
                                                       y_val),
                                      use_multiprocessing=True,
                                      max_queue_size=96,
                                      workers=12,
                                      steps_per_epoch=conf.steps_per_epoch,
                                      epochs=conf.epochs if not developing
                                      else 2,
                                      callbacks=setup_clbks(
                                          conf,
                                          conf.log_dir),
                                      verbose=1)
    else:
        X_train = np.asarray([conf.data_loader(x) for x in dataset.train[0]])
        y_train = dataset.train[1]
        history = model.fit(x=X_train, y=y_train,
                            batch_size=conf.batch_size,
                            validation_data=(X_val, y_val),
                            epochs=conf.epochs if not developing
                            else 2,
                            callbacks=setup_clbks(
                                conf,
                                conf.log_dir),
                            verbose=1)
    pickle.dump(history,
                open(os.path.join(conf.log_dir,
                                  f'history_{conf}.pkl'), 'wb'),
                protocol=3)
    return history


def main(conf: Config):
    model = build_model(conf)
    conf.model_name = 'mit'
    logm(f'Current configuration is:\n{repr(conf)}',
         cur_frame=currentframe(), mtype='I')
    model.summary()

    setup_dirs(conf)
    if TRAIN:
        logm(f'Start train: {str(conf)}',
             cur_frame=currentframe(), mtype='I')
        with Timer() as t:
            train(model, conf, developing=DEVELOPING)

        logm(f'End train: total time taken: {str(t.interval)}',
             cur_frame=currentframe(), mtype='I')
    if SAVE_MODEL:
        logm(f'Saving model: {str(conf.model_name)} at {conf.model_location}',
             cur_frame=currentframe(), mtype='I')
        model.save(conf.model_location + '.h5')
    if EVALUATE:
        logm(f'Start evaluation: {str(conf)}',
             cur_frame=currentframe(), mtype='I')

        test_paths, test_labels = parse_csv(conf.test_data_csv,
                                            conf.data_path)
        if DEVELOPING:
            logm('Developing is set as true: limiting size of dataset',
                 cur_frame=currentframe(), mtype='I')
            paths_labels = list(zip(test_paths, test_labels))
            random.shuffle(paths_labels)
            test_paths, test_labels = zip(*paths_labels)
            test_paths = test_paths[:10]
            test_labels = test_labels[:10]

        logm(f'Get test data: total of {len(test_labels)}',
             cur_frame=currentframe(), mtype='I')
        if not TRAIN:
            logm(f'Loading model from {conf.model_location}',
                 cur_frame=currentframe(), mtype='I')
            if conf.model_location is None:
                logm('Model location is None: '
                     'Load the configuration file of a valid trained '
                     'model', cur_frame=currentframe(), mtype='E')
                raise ValueError('Not valid configuration file for evaluation')
            model = load_model(conf.model_location + '.h5')
        logm(f'Start evaluation of {str(conf)}',
             cur_frame=currentframe(), mtype='I')
        test_dataset = TestDataset(test_paths, test_labels,
                                   name=conf.dataset_name,
                                   num_classes=conf.num_classes)
        with Timer() as t:
            evaluate(model, conf, test_dataset, 'final')
        logm(f'End evaluation: total time taken: {str(t.interval)}',
             cur_frame=currentframe(), mtype='I')
        if EVALUATE_BEST_MODEL:
            logm('Evaluating best model', cur_frame=currentframe(), mtype='I')
            logm(f'Loading model from {conf.model_checkpoint_location}',
                 cur_frame=currentframe(), mtype='I')
            model = load_model(conf.model_checkpoint_location +
                                            '.h5')
            logm(f'Start evaluation of {str(conf)} (best model)',
             cur_frame=currentframe(), mtype='I')
            with Timer() as t:
                evaluate(model, conf, test_dataset, 'best')
            logm('End evaluation (best model): total time taken: '
                 f'{str(t.interval)}', cur_frame=currentframe(), mtype='I')


if __name__ == '__main__':
    with open(LOG_FILE, 'a') as log_file:
        log_file.write('\n---------------------------------------------\n')
        log_file.write(f'LOG: {__file__}: {datetime.now()}\n')
    try:
        if (not TRAIN and EVALUATE) or LOAD_CONF:
            if CONF_LOCATION is None:
                raise Exception('Must set CONF_LOCATION')
            conf = Config.frompicke(CONF_LOCATION)
        else:
            conf = Config(conf_name=f'{__file__}_{SAMPLING_RATE}_'
                          f'{MAX_SECONDS_PER_RUN}_{STEPS_PER_EPOCH}',
                          data_loader=wav_to_specdata,
                          use_tb_embeddings=TB_EMBEDDINGS)
        main(conf)
    except Exception as err:
        logm(f'FATAL ERROR: {str(err)}', cur_frame=currentframe(),
             mtype='E')
        os.remove(str(conf) + '.pkl')
        raise err
