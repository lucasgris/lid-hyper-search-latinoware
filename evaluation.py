from keras.models import Model

from common.config import Config, REPORT_FILE
from common.datasets import Dataset
from common.generator import Generator


def evaluate(model: Model, conf: Config, dataset: Dataset, model_name=None):
    """
    Simply evaluate a model and appends the result to a report file.
    See common.config.REPORT_FILE.
    
    Args:
        model (Model): a keras model to evaluate.
        conf (Config): a config object.
        dataset (Dataset): a dataset to use as test.
        model_name (str, optional): A model name. Defaults to None, will use
            the config model name.
    """
    test_paths, test_labels = dataset.test
    test_gen = Generator(test_paths, test_labels,
                         batch_size=1,
                         loader_fn=conf.data_loader)
    score = model.evaluate_generator(generator=test_gen, verbose=1)
    with open(REPORT_FILE, 'a') as report:
        model_name = conf.model_name if model_name is None else model_name
        report.write(f'{model_name},{conf},{str(dataset)},{score[0]},'
                     f'{score[1]}\n')
