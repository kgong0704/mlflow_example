import random
import jieba
from classifier.utils.new_dct import key_words_dct
from classifier.affairs_classifier.RegexAnnotation import regex_annotation
from classifier.utils.environmental_variables import STOP_WORDS_PATH
from classifier.utils.jieba_additional_words import JIEBA_ADDITIONAL_WORDS
from classifier.utils.environmental_variables import MODEL_ROOT, DATA_SET_PATH
import os


class PreProcess:
    def __init__(self, mode):
        self.mode = mode

    @staticmethod
    def get_stopwords_list(stop_word_file):
        stop = set()
        with open(stop_word_file, 'r', encoding='gbk') as f:
            lines = f.readlines()  # lines是list类型
            for line in lines:
                lline = line.strip()  # line 是str类型,strip 去掉\n换行符
                stop.add(lline)
        return stop

    @staticmethod
    def seg_sentence(sentence, stop_words):
        # add words for jieba
        seg_sentence = jieba.cut(sentence.strip(), cut_all=False)
        out_str = ''
        for word in seg_sentence:
            if word not in stop_words:
                if word != '\t':
                    out_str += word
                    out_str += " "
        return out_str

    @staticmethod
    def drop_negative(dct_data, negative_sampling_factor):
        for file_name, sentences in dct_data.items():
            processed_positive = []
            processed_negative = []

            for s in sentences:
                _, label, seg_sentence, sentence = s.split('  ', 3)
                if label == '__label__负样本':
                    processed_negative.append(file_name + '  ' + label + '  ' + seg_sentence + '  ' + sentence)
                else:
                    processed_positive.append(file_name + '  ' + label + '  ' + seg_sentence + '  ' + sentence)
            random.shuffle(processed_positive)
            processed = processed_positive + processed_negative[:int(negative_sampling_factor*len(processed_positive))]
            dct_data[file_name] = processed
        return dct_data

    def pre_process_json(self, raw_data, test_sample_percentage):
        dct_data = raw_data
        for _, word_list in JIEBA_ADDITIONAL_WORDS.items():
            [jieba.add_word(w) for w in word_list]

        stopwords = self.get_stopwords_list(STOP_WORDS_PATH)

        for file_name, sentences in dct_data.items():
            all_data = []
            for s in sentences:
                # s[1] -> the original sentence
                label, sentence = s[0].split(',', 1)
                seg_sentence = self.seg_sentence(sentence, stopwords)
                # filter out short sentences
                if len(seg_sentence.split(' ')) < 2: continue
                all_data.append(file_name + '  ' + label + '  ' + seg_sentence + '  ' + s[1])
            dct_data[file_name] = all_data

        items = list(dct_data.items())
        random.shuffle(items)

        dct_test = {}
        total_size, cnt = len(items), 0
        for k, v in items:
            if cnt < int(total_size * test_sample_percentage):
                dct_test[k] = v
                del dct_data[k]
                cnt += 1
        return dct_data, dct_test

    def split_and_write(self, processed_dct, dataset_root, *args):
        total_num, cnt = len(processed_dct), -1
        training_set = open(os.path.join(dataset_root, 'train.txt'), 'w', encoding='utf-8')
        validating_set = open(os.path.join(dataset_root, 'valid.txt'), 'w', encoding='utf-8')

        items = list(processed_dct.items())
        random.shuffle(items)

        kw = []
        for k, _ in key_words_dct.items(): kw.append(k)
        kw.append('负样本')

        for file_name, sentences in items:
            for s in sentences:
                _, label, _ = s.split('  ', 2)

                if self.mode == 'multi_label':
                    condition = label[9:] in kw
                else:
                    condition = len(label.split(' ')) == 1 and label[9:] in kw

                if condition:
                    if cnt <= int(total_num * 0.8):
                        training_set.write(s)
                        training_set.write('\n')
                    else:
                        validating_set.write(s)
                        validating_set.write('\n')
        training_set.close()
        validating_set.close()

        if args:
            dct_test = args[0]
            testing_set = open(os.path.join(dataset_root, 'train.txt'), 'w', encoding='utf-8')
            for file_name, sentences in dct_test.items():
                for s in sentences:
                    _, labels, _ = s.split('  ', 2)
                    label = labels.split(' ')
                    valid = [0]*len(label)
                    for i in range(len(label)):
                        if label[i][9:] in kw: valid[i] = 1
                    if set(valid) == {1}:
                        testing_set.write(s)
                        testing_set.write('\n')
            testing_set.close()


def generate_dataset(conf):
    mode = 'multi_label' if conf.get('loss_function') in ('ova', 'ns') else 'single_label'
    pp = PreProcess(mode)

    pre_processed_data = regex_annotation(conf.get('url_list'))
    dct_train, dct_test = pp.pre_process_json(pre_processed_data, conf.get('val_set_ratio'))
    dct_train = pp.drop_negative(dct_train, conf.get('negative_ratio_to_positive'))
    pp.split_and_write(dct_train, DATA_SET_PATH, dct_test)
