import fasttext
import pandas as pd
import os
from classifier.utils.environmental_variables import MODEL_ROOT, DATA_SET_PATH


def train(conf, model_path):
    def rewrite_test_set(test_set_path):
        path_for_train = 'tmp.txt'
        data = open(test_set_path, 'r', encoding='utf-8')
        for_train = open(path_for_train, 'w', encoding='utf-8')
        lines = data.readlines()
        cnt0 = 0
        cnt1 = 1
        for l in lines:
            _, label, text, ori_text = l.split('  ', 3)
            if label == '__label__负样本':
                cnt0 += 1
            else:
                cnt1 += 1
            for_train.write(label + ' ' + text)
            for_train.write('\n')
        return path_for_train

    train_set = rewrite_test_set(os.path.join(DATA_SET_PATH, 'train.txt'))

    model = fasttext.train_supervised(input=train_set, lr=conf.get('learning_rate'), epoch=conf.get('epochs'),
                                      wordNgrams=conf.get('wordNgram'), loss=conf.get('loss_function'),
                                      dim=conf.get('model_dimension'))
    os.remove(train_set)
    model.save_model(model_path)


def evaluate_performance(model, validation_file):
    validations_swr = open(validation_file, 'r', encoding='utf-8').readlines()
    res = {}
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(validations_swr)):
        if validations_swr[i] != '\n':
            file_name, true_labels, sentence_to_predict, ori_sen = validations_swr[i].split('  ', 3)
            if file_name not in res:
                res[file_name] = []
            sentence_to_predict = sentence_to_predict.replace('\n', '')

            pred_label = model.predict(sentence_to_predict, threshold=0.5, k=-1)

            # define ova confusion matrix
            if not pred_label[0]:
                if true_labels == '__label__负样本':
                    tn += 1
                else:
                    fn += 1

            else:
                if true_labels == '__label__负样本':
                    if set(pred_label[0]) == {'__label__负样本'}:
                        tn += 1
                    else:
                        fp += 1
                else:
                    if set(pred_label[0]) == set(true_labels.split(' ')):
                        tp += 1

                    elif pred_label[0][0] == '__label__负样本':
                        fn += 1
                    else:
                        fp += 1
    print(tp, tn, fp, fn)
    print('Precision:  ' + str(round(tp/(tp + fp), 2)), 'Accuracy: ' + str(round((tp + tn) / (tp + tn + fp + fn), 2)),
          'Recall: ' + str(round(tp/(tp + fn), 2)))


def get_predict_label_per_pdf(model, test_path):
    """
    get list of labels per pdf file
    :param test_path:
    :return:
    """
    cur_file_name = 'not'
    cur_file_labels = {'predict': set(),
                       'ground_truth' : set()}
    total_res = []

    res_matrix = []

    validations_swr = open(test_path, 'r', encoding='utf-8').readlines()
    for i in range(len(validations_swr)):
        if validations_swr[i] != '\n':
            file_name, true_labels, sentence_to_predict, ori_sen = validations_swr[i].split('  ', 3)
            if cur_file_name and file_name != cur_file_name:
                # append last file res to total
                total_res.append(cur_file_labels)
                cur_file_name = file_name
                cur_file_labels = {'predict': set(),
                                   'ground_truth': set()}
                # update current file info
            predict = model.predict(sentence_to_predict, threshold=0.4, k=-1)
            true_label = true_labels.split(' ')
            for p in predict[0]:
                cur_file_labels['predict'].add((p, file_name))
            for t in true_label:
                cur_file_labels['ground_truth'].add((t, file_name))
    total_res.append(cur_file_labels)

    all_correct, total = 0, 0
    tp, fp, fn = 0, 0, 0
    with open('test.txt', 'w') as f:
        for res in total_res:
            if (not res['ground_truth'] - res['predict']):
                all_correct += 1
                tp += len(res['ground_truth'])
            else:
                intersected = res['ground_truth'].intersection(res['predict'])
                miss = res['ground_truth'] - res['predict']
                extra = res['predict'] - res['ground_truth']
                for m in miss:
                    if m[0] != '__label__负样本':
                        f.write(str(m) + '\n')
                tp += len(intersected)
                fn += len(miss)
                fp += len(extra)
            total += 1
        print('全部预测正确：', all_correct, '全部文档数量：', total, '文档级别召回率： ', all_correct/total, '\n', tp, fp, fn, '\n', 'recall: ', tp/(tp + fn),
              ' precision: ', tp/(tp + fp))

    name_list = []
    predict = []
    regex = []

    total_res.pop(0)
    for res in total_res:
        name, pred, reg = '', '', ''
        for _, v in enumerate(res['predict']):
            if name != '': name = v[1]
            if v[0] != '__label__负样本':
                pred += ' ' + v[0][9:]
        for _, v in enumerate(res['ground_truth']):
            name = v[1]
            if v[0] != '__label__负样本':
                reg += ' ' + v[0][9:]

        name_list.append(name)
        predict.append(pred)
        regex.append(reg)

    from classifier.utils.new_dct import key_words_dct
    with pd.ExcelWriter('重大事项分类模型、标注对比.xlsx', engine='xlsxwriter') as writer:
        dct_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in key_words_dct.items()]))
        dct_df.to_excel(writer, sheet_name='关键词列表')

        data = {'文件下载url': name_list, '模型预测': predict, '正则匹配': regex}
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='data')


def label_performances_to_csv(model):
    def _evaluate_confusion_matrix_for_labels(model, label, validation_file):
        validations = open(validation_file, 'r', encoding='utf-8').readlines()
        tp, tn, fp, fn = 0, 0, 0, 0

        for line in validations:
            if line != '\n':
                true_labels, sentence = line[1:].split('  ', 1)
                true_labels = true_labels.split(' ')
                sentence = sentence.replace('\n', '')
                predict_labels = model.predict(sentence, k=len(true_labels))[0]

                if label in predict_labels:
                    for t in true_labels:
                        if t in predict_labels:
                            tp += 1
                        else:
                            fp += 1
                else:
                    if label in true_labels:
                        fn += 1
                    else:
                        tn += 1

        return tp, tn, fp, fn, (round((tp) / (tp + fn), 2) if (tp + fp) > 0 else 'N/A'), round(
            (tp + tn) / (tp + tn + fp + fn), 2)

    labels = model.get_labels()
    df_raw = [[None for x in range(7)] for y in range(len(labels))]
    for i in range(len(labels)):
        res = _evaluate_confusion_matrix_for_labels(model, labels[i], 'dataset/blended_swr.test.txt')
        df_raw[i][0] = model.get_labels()[i][9:]

        for j in range(len(res)):
            df_raw[i][j + 1] = res[j]
    rez = [[df_raw[j][i] for j in range(len(df_raw))] for i in range(len(df_raw[0]))]

    data = {'label': rez[0], 'tp': rez[1], 'tn': rez[2], 'fp': rez[3], 'fn': rez[4], 'acc': rez[5], 'rec': rez[6]}
    df = pd.DataFrame(data)
    df.to_csv('performance/individual_label_performance.csv', encoding='utf_8_sig')
