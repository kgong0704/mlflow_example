import re
from classifier.utils.new_dct import key_words_dct
import collections
import pandas as pd
import requests


class RegexAnnotation:
    def __init__(self):
        pass

    @staticmethod
    def write_json_to_text(json_data):
        incomplete_text = ''
        paragraphs = []
        s1 = json_data
        pages = s1['pages']
        cnt_front, cnt_back = 0, 0
        for p in pages:
            paragraphObject = p['paragraphs']
            for i in range(len(paragraphObject)):
                if 'texts' in paragraphObject[i]:
                    if 'nextPageParagraph' in paragraphObject[i]:
                        incomplete_text += paragraphObject[i]['text']
                    elif 'prevPageParagraph' in paragraphObject[i]:
                        incomplete_text += paragraphObject[i]['text']
                        paragraphs.append(incomplete_text)
                        incomplete_text = ''
                    else:
                        paragraphs.append(paragraphObject[i]['text'])
        return paragraphs

    @staticmethod
    def cut_to_sentences(para):
        para = re.sub('([。！？\?；])([^”’])', r"\1\n\2", para)
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        para = para.rstrip()
        return para.split("\n")

    @staticmethod
    def match_sentences_regex_ver(sentence, dct):
        matched = []
        matched_keyword = []
        # use black list to blackout matched tags (not keywords)
        black_list = []
        get_new = True
        while get_new:
            get_new = False
            for k, v in dct.items():
                if k not in black_list:
                    sentence_to_check = sentence[:]
                    for key_word in v:
                        if re.search(r'{}'.format(key_word), sentence_to_check):
                            matched.append(k)
                            matched_keyword.append(key_word)
                            black_list.append(k)
                            get_new = True
                            # key word matched => sentences[sent_index][start:end+1]
        return matched, matched_keyword

    @staticmethod
    def load_browser_info(upload_files_list, num_of_files):
        file_info = []
        for i in range(num_of_files):
            file = upload_files_list[i]
            file_info.append((file['material_id'], file['name']))
        return file_info

    @staticmethod
    def load_result(key_word_list):
        true_res = {}
        file_matched_in_target = set()
        data = pd.read_excel('工作簿1.xlsx')
        data_list = data.values.tolist()
        for d in data_list:
            t = re.split('/', d[2].strftime("%m/%d/%Y, %H:%M:%S"))
            key = d[0] + re.split('\\.', d[1])[0] + t[2][:4] + t[0] + t[1]
            if key in true_res:
                true_res[key].append(d[4].replace('\xa0', ''))
            else:
                true_res[key] = [d[4].replace('\xa0', '')]
        # make the result to list
        for k, tags in true_res.items():
            res_list = [0]*len(key_word_list)
            for v in tags:
                # if v in key_word_list:
                #     # if v == TAG_TO_CHECK:
                #     #     file_matched_in_target.add(k)
                res_list[key_word_list.index(v)] = 1
            true_res[k] = res_list
        return true_res, file_matched_in_target

    def count_apperance(self, all_matches):
        Cntr = collections.Counter(all_matches).most_common()
        dct_to_print = {}

        for k, v in key_words_dct.items():
            dct_to_print[k] = []
            for c in Cntr:
                if c[0] in v:
                    dct_to_print[k].append(c)

        df = pd.DataFrame()
        for k, v in dct_to_print.items():
            sum = 0
            list = []
            for _v in v:
                sum += _v[1]
                list.append(_v[0] + ' : ' + str(_v[1]))
            df[k + ' : ' + str(sum)] = pd.Series(list)
        df.to_csv('performance/tag_match_performance.csv', encoding='utf_8_sig')

    def write_alljson_to_dataset(self, name, data, myjson, all_match_in_dataset, new_dct):
        def regex_cleanup(vTEXT):
            vTEXT = re.sub(r' ', '', vTEXT, flags=re.MULTILINE)
            # vTEXT = re.sub(r'.', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'一|二|三|四|五|六|七|八|九|Ｏ', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'股份有限公司|有限公司', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'全国中小企业股份转让系统', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\年(.*?)\日', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\年(.*?)\月', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\月(.*?)\日', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\(.*?\)', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\（.*?\）', '', vTEXT, flags=re.MULTILINE)
            # vTEXT = re.sub(r'\《.*?\》', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\“.*?\”', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'[a-zA-Z]', '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(
                r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',
                '', vTEXT, flags=re.MULTILINE)
            vTEXT = re.sub(r'\d+', '', vTEXT, flags=re.MULTILINE)
            return (vTEXT)

        k_s = {}
        # todo: modify to load locally
        paragraphs = self.write_json_to_text(data)
        for p in paragraphs:
            sentences = self.cut_to_sentences(p)
            paragraph_label = []
            for i in range(len(sentences)):
                cleaned_up = regex_cleanup(sentences[i])
                match, matched_keyword = self.match_sentences_regex_ver(cleaned_up, new_dct)
                if match:
                    k_s[cleaned_up] = (match, name, sentences[i])
                else:
                    k_s[cleaned_up] = (['负样本'], name, sentences[i])
                paragraph_label.append(match)
                for k in matched_keyword: all_match_in_dataset.append(k)

        # fasttext data set format
        for k, v in k_s.items():
            t = ''
            for p in set(v[0]):
                if p != '负样本':
                    t = t + '__label__' + p + ' '
                else:
                    t = t + '__label__负样本 '
            t += ', ' + k

            if v[1] not in myjson:
                myjson[v[1]] = []
            myjson[v[1]].append((t, v[2]))
        return myjson, all_match_in_dataset


def regex_annotation(urls):
    f = RegexAnnotation()
    all_match_in_dataset = []

    myjson = {}
    for url in urls:
        data = requests.get(url).json()
        name = data['fileUrl']
        f.write_alljson_to_dataset(name, data, myjson, all_match_in_dataset, key_words_dct)
    return myjson


# if __name__ == '__main__':
#     raw_foler_path = 'raw_json_data/85'
#     folder_name = os.path.basename(raw_foler_path)
#     regex_annotation(raw_foler_path, os.path.join('dataset/raw', '{}.json'.format(folder_name)))
