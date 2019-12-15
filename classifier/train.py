import fasttext
import os
from classifier.affairs_classifier.file_handler import generate_dataset
from classifier.affairs_classifier.train_and_evaluation import train, evaluate_performance
from classifier.utils.environmental_variables import MODEL_ROOT, DATA_SET_PATH
import mlflow
import sys
import json
from datetime import date


class Train:
    def __init__(self):
        pass

    @staticmethod
    def train():
        # load data and write to dictionary form
        def training(conf):

            generate_dataset(conf)

            model_path = os.path.join(MODEL_ROOT, conf.get('model_name'))
            train(conf, model_path)
            model = fasttext.load_model(model_path)
            evaluate_performance(model, os.path.join(DATA_SET_PATH, 'test.txt'))

        def generate_model_name(data_set_name, loss_function):
            today = date.today()
            d1 = today.strftime("%d%m%Y")
            name = '{0}_{1}_{2}.bin'.format(d1, data_set_name, loss_function)
            return name

        parameter_config = json.loads(sys.argv[1])
        raw_data_set = json.loads(sys.argv[2])
        conf = {
            # todo: need to discuss format of url list
            'url_list': 'affairs_classifier/urllist.json',
            'negative_ratio_to_positive': 1.5,
            'learning_rate': 0.5,
            'epochs': 100,
            'wordNgram': 1,
            'model_dimension': 10,
            'val_set_ratio': 0.2,
            'model_name': 'my_fasttext.bin',
            'loss_function': 'ova',
        }

        model_name = generate_model_name(raw_data_set.get('name'), conf.get('loss_function'))
        parameter_config['model_name'] = model_name
        parameter_config['url_list'] = raw_data_set.get('urls')

        conf.update(parameter_config)
        with mlflow.start_run():
            training(conf)


if __name__ == '__main__':
    p = Train()
    p.train()
