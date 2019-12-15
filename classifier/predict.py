from __future__ import unicode_literals, print_function
import os
import fasttext
from classifier.affairs_classifier.file_handler import PreProcess
from classifier.affairs_classifier.RegexAnnotation import RegexAnnotation
from classifier.utils.split_sentences import get_paragraph, get_splited_sentence
from classifier.utils.environmental_variables import MODEL_ROOT, STOP_WORDS_PATH
import requests
import re


def _load_pyfunc(path):
    return Predict(path=path)


class Predict:
    def __init__(self, path):
        self.path = path

    # mlflow process the request through a flask app and use pd.read_json to preprocess raw data, thus input transform
    # from python dictionary to pandas DataFrame
    @staticmethod
    def predict(data_from_mlflow):
        data_dict = data_from_mlflow.to_dict("records")[0]
        json_url_list = data_dict['url']
        model_name = data_dict['model_name']
        data = requests.get(json_url_list).json()

        file_name = data['_id']['$oid']
        paras, params = get_paragraph(data['pages'])
        sentences = get_splited_sentence(paras)
        model_names = os.listdir(MODEL_ROOT)

        if model_name in model_names:
            model_to_use = model_name
        else:
            # backup model to use
            model_to_use = '2000_ova.bin'

        # get model mode from its name
        mode_type = re.findall(r'[^_]+$', model_to_use)[0][:-4]

        # use fasttext to process the sentences
        process = PreProcess(mode_type)
        model = fasttext.load_model(os.path.join(MODEL_ROOT, model_to_use))
        stopwords = process.get_stopwords_list(STOP_WORDS_PATH)

        res_list = []
        labels_in_curr_file = set()
        sentence_info = []
        for s in sentences:
            seg_sentence = process.seg_sentence(s['text'], stopwords)
            label = model.predict(seg_sentence, threshold=0.5)
            if label[0]:
                if label[0][0] == '__label__负样本':
                    continue

                sentence_info.append({'label': label[0][0][9:], 'texts': s['text'], 'x': s['x'], 'y': s['y'],
                                      'w': s['w'], 'h': s['h']})
                labels_in_curr_file.add(label[0][0][9:])

        res_list.append({'$oid': file_name, 'labels': list(labels_in_curr_file), 'sentences_info': sentence_info})
        return res_list


if __name__ == '__main__':
    from pandas import DataFrame

    data = DataFrame(columns=["url", "model_name"],
            data=[["http://abc-crawler.oss-cn-hangzhou.aliyuncs.com/texts/52dedce05b5c6142f37f35819ac5078c4b8f66c40275e0dc440968bb572bf4d1/all.json", "jzy160_model_hs.bin"]])

    get_traindata = Predict("")
    api_data = get_traindata.predict(data)
    print(api_data)
