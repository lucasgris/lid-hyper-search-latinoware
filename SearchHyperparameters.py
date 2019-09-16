#!/usr/bin/env python
# coding: utf-8

# In[1]:


from inspect import currentframe, getframeinfo
from time import time
from datetime import datetime
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model, save_model, Model
from keras import backend as K 
import os
import keras
import random
import numpy as np
import pprint
import sys

from common.config import *
from common.util import logm, parse_csv, Timer
from common.dataloader import (wav_to_specdata, normalize, spec_load_and_rshp,
                               spec_save)
from common.datasets import Dataset, TestDataset
from common.generator import Generator, Heap
from common.callbacks import TimedStopping
from evaluation import evaluate
from model import build_model
from params import (params, can_shake, create_space,
                    params_keys_comma_separated, params_values_comma_separated)


# In[2]:


DEVELOPING = True
RUNS = int(sys.argv[1])
TIME_LIMIT = sys.argv[2] if len(sys.argv) > 2 else None # "25/09/2019 12:00:00"


# In[3]:


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
                setup_mc=False,
                setup_ts=True,
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


# In[4]:


def test_space(spaces, remove_bad_topologies=True):
    pp = pprint.PrettyPrinter(indent=4)
    for i, space in enumerate(spaces):
        logm(f'Testing space [{i+1} of {len(spaces)}]',
             cur_frame=currentframe(), mtype='I')
        pp.pprint(space)
        try:
            K.clear_session()
            model = build_model(conf, space, input_shape=(SPEC_SHAPE_HEIGTH,
                                                          SPEC_SHAPE_WIDTH,
                                                          CHANNELS))
        except ValueError as err:
            logm(f'Failed when building the model: {str(err)} ',
                 cur_frame=currentframe(), mtype='I')
            if remove_bad_topologies:
                del space
            continue
    return spaces


# In[5]:


def train(model: Model, conf: Config, batch_size, developing=False):
    logm(f'Running train for {conf}', cur_frame=currentframe(),
         mtype='I')
    train_paths, train_labels = parse_csv(conf.train_data_csv, conf.data_path)
    val_paths, val_labels = parse_csv(conf.eval_data_csv, conf.data_path)
    if developing:
        logm('Developing is set as true: limiting size of dataset',
             cur_frame=currentframe(), mtype='I')
        train_paths_labels = list(zip(train_paths, train_labels))
        random.shuffle(train_paths_labels)
        train_paths, train_labels = zip(*train_paths_labels)
        train_paths = train_paths[:100]
        train_labels = train_labels[:100]
        val_paths = val_paths[:10]
        val_labels = val_labels[:10]
        epochs = 2
        conf.steps_per_epoch = 10

    train_dataset = Dataset(train_paths, train_labels,
                            name=conf.dataset_name + '[TRAIN]',
                            num_classes=NUM_CLASSES)
    val_dataset = Dataset(val_paths, val_labels,
                          name=conf.dataset_name + '[VALIDATION]',
                          num_classes=NUM_CLASSES)
    
    epochs = int(len(train_dataset)//(batch_size*conf.steps_per_epoch)) + 1
    logm(f'Calculated number of epochs to process all data at least 1 time: {epochs}',
         cur_frame=currentframe())

    logm(f'Loading validation data...', cur_frame=currentframe(),
         mtype='I')
    X_val = np.asarray([conf.data_loader(x) for x in val_dataset()[0]])
    y_val = val_dataset()[1]
    logm(f'Loading validation data... Done', cur_frame=currentframe(),
         mtype='I')
    logm(f'Validation data shape is {X_val.shape}',
         cur_frame=currentframe(),
         mtype='I')
    if conf.use_generator:
        logm('Using generator', cur_frame=currentframe(), mtype='I')
        train_gen = Generator(paths=train_dataset()[0],
                              labels=train_dataset()[1],
                              loader_fn=conf.data_loader,
                              batch_size=int(batch_size),
                              heap=setup_heap(conf) if conf.use_heap else None,
                              expected_shape=(SPEC_SHAPE_HEIGTH,
                                              SPEC_SHAPE_WIDTH,
                                              CHANNELS))
        history = model.fit_generator(generator=train_gen,
                                      validation_data=(X_val,
                                                       y_val),
                                      use_multiprocessing=True,
                                      max_queue_size=96,
                                      workers=12,
                                      steps_per_epoch=conf.steps_per_epoch,
                                      epochs=epochs,
                                      callbacks=setup_clbks(conf),
                                      verbose=1)
    else:
        X_train = np.asarray([conf.data_loader(x) for x in train_dataset()[0]])
        y_train = train_dataset()[1]
        history = model.fit(x=X_train, y=y_train,
                            batch_size=int(batch_size),
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            callbacks=setup_clbks(conf),
                            verbose=1)
    return history, model


def random_search(conf: Config, search_space):
    logm(f'Running random search for {conf}', cur_frame=currentframe(),
         mtype='I')
    logm(f'Created report file at {conf.report_file}',
         cur_frame=currentframe(), mtype='I')

    with open(conf.report_file, 'w') as output:
        output.write(f'run,{params_keys_comma_separated()},validation_acc,'
                     'time_taken\n')

    for i, space in enumerate(search_space):
        logm(f'Running random search on space {space} - {i+1} of '
             f'{len(search_space)}', cur_frame=currentframe(), mtype='I')

        if conf.time_limit is not None and datetime.now() > conf.time_limit:
            logm('Time limit reached: end random search',
                 cur_frame=currentframe(), mtype='I')
            return
        logm('Buiding model', cur_frame=currentframe(), mtype='I')
        try:
            K.clear_session()
            model = build_model(conf, space, input_shape=(SPEC_SHAPE_HEIGTH,
                                                          SPEC_SHAPE_WIDTH,
                                                          CHANNELS))
        except ValueError as err:
            logm(f'Error when building the model: {str(err)} ',
                 cur_frame=currentframe(), mtype='E')
            continue
        model.summary()
        with Timer() as t:
            result, model = train(model, conf, space['batch_size'],
                                  developing=DEVELOPING)
        time_taken = t.interval
    
        validation_acc = np.amax(result.history['val_acc'])
        with open(conf.report_file, 'a') as output:
            output.write(f'{conf.run},{params_values_comma_separated(space)},'
                         f'{validation_acc},{time_taken}\n')
        conf.run += 1
        with open(os.path.join(conf.log_dir, f'hist_acc_{conf.run}.csv'),
                               'w') as output:
            output.write(f'epoch,val_acc\n')
            for itr, val_acc in enumerate(result.history['val_acc']):
                output.write(f'{itr+1},{val_acc}\n')
        if SAVE_MODEL:
            model_path = os.path.join(conf.models_dir, conf.model_name +
                                      f'_{conf.run}.h5')
            logm(f'Saving model at {model_path}',
                 cur_frame=currentframe(), mtype='I')
            model.save(model_path)


# In[6]:


def main(conf: Config):
    logm('>>> Started random hyperparameter search <<<')

    conf.model_name = f'random_search_{TIME_NOW}'
    logm(f'Current configuration is:\n{repr(conf)}',
         cur_frame=currentframe(), mtype='I')

    setup_dirs(conf)
    logm('Creating space', cur_frame=currentframe())
    space = create_space(RUNS)
    space = test_space(space, remove_bad_topologies=True)
    logm(f'Start random search: {str(conf)}', cur_frame=currentframe())
    with Timer() as t:
        random_search(conf, space)

    logm(f'End random search: total time taken: {str(t.interval)}',
         cur_frame=currentframe(), mtype='I')
    
    logm('End evaluation (best model): total time taken: '
            f'{str(t.interval)}', cur_frame=currentframe(), mtype='I')


# In[ ]:


with open(LOG_FILE, 'a') as log_file:
        log_file.write(f'\n{"="*80}\n')
        log_file.write(f'LOG SearchHyperparameters.ipynb:{datetime.now()}\n')
conf = Config(params=params,
              conf_name=f'SearchHyperparameters.ipynb_{SAMPLING_RATE}_'
              f'{MAX_SECONDS_PER_RUN}_{STEPS_PER_EPOCH}',
              data_loader=wav_to_specdata,
              use_tb_embeddings=TB_EMBEDDINGS)
conf.max_seconds_per_run = MAX_SECONDS_PER_RUN
if TIME_LIMIT is not None:
    time_limit = datetime.strptime(TIME_LIMIT, "%d/%m/%Y %H:%M:%S")
    conf.time_limit = time_limit
try:
    main(conf)
except Exception as err:
    logm(f'FATAL ERROR: {str(err)}', cur_frame=currentframe(),
         mtype='E')
    conf.delete_file()
    raise err


# In[ ]:




