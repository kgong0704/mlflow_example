# coding:utf-8
'''
@Auther: qwang
@Time: 2019-11-03 15:57
@Filename: test_splite_sentence
'''
import re
import os
import logging
import nltk
#nltk.download("punkt")
from nltk.tokenize import sent_tokenize

ROOTPATH = os.path.dirname(os.path.dirname(__file__))

if __name__ != "__main__":
    from mlflow.pyfunc.__init__ import _logger

    log = _logger.getChild(__file__.replace(ROOTPATH,"",1))
else:
    log = logging.getLogger(__name__)


def get_paragraph(pages):
    """
    将文本转化为独立的parapraph　同时记录文本的左右边界和字体大小信息
    :param pages:
    :return:
    """

    def add_last_space(my_para):
        if len(my_para['char_area_list']) < len(my_para['text']):  # 补上末尾的空格
            space_count = len(my_para['text']) - len(my_para['char_area_list'])
            last_char = my_para['char_area_list'][-1]
            my_para['char_area_list'] += [
                {"x": last_char['x'] + i * 2., "y": last_char['y'], "w": 2., "h": last_char['h']} for
                i in range(space_count)]

    if not pages:
        return
    my_paras = []
    font_size_dict = {}
    left_x = 9999.0
    right_x = 0.0
    for page in pages:
        page_index = page['pageIndex']
        for para in page['paragraphs']:
            my_para = {}
            my_para['page_index'] = page_index
            my_para['text'] = para['text']
            my_para['pid'] = len(my_paras)
            my_para['x'], my_para['y'], my_para['w'], my_para['h'] =\
                para['area']['x'], para['area']['y'], para['area']['w'], para['area']['h']
            if para['area']['x'] < left_x:
                left_x = para['area']['x']

            if (para['area']['x']+para['area']['w']) > right_x:
                right_x = para['area']['x']+para['area']['w']

            my_para['char_area_list'] = []
            p = 0  # 上一文本块的结尾
            max_font_size = 0
            for content in para['texts']:
                font_size = content['font_size']
                if font_size not in font_size_dict:
                    font_size_dict[font_size] = 0
                font_size_dict[font_size] += 1

                if font_size > max_font_size:
                    max_font_size = font_size
                x, y, w, h = content['x'], content['y'], content['w'], content['h']
                text_re_s = re.escape(content['text'])
                tar = re.search(text_re_s, para['text'][p:])
                space_count = tar.start() if tar else None
                p += tar.end() if tar else len(re.sub(r'\s', '', content['text']))
                if space_count:
                    last_char = my_para['char_area_list'][-1] if my_para['char_area_list'] else \
                        {"x": my_para['x'], "y": y, "w": 1., "h": h}  # 以空格开头
                    space_width = max(
                        (content['char_left'][0] - (last_char['x'] + last_char['w'])) / space_count, 2.)
                    my_para['char_area_list'] += [{"x": last_char['x'] + last_char['w'] + i * space_width,
                                                   "y": last_char['y'], "w": space_width, "h": last_char["h"]}
                                                  for i in range(space_count)]
                my_para['char_area_list'] += [{"x": l, "y": y, "w": r - l, "h": h} for
                                             r, l in zip(content['char_right'], content['char_left'])]

            my_para['font_size'] = max_font_size
            add_last_space(my_para)
            # assert len(my_para['char_area_list']) == len(my_para['text'])
            my_paras.append(my_para.copy())
            del my_para
    params = {'font_size_dict': font_size_dict, 'left_x': left_x, 'right_x': right_x}

    return my_paras, params
def chars2bbox(char_area_list):
    """
    :param char_area_list: [{"x": ..., "y":..., ...}, ...]
    :return:
    """
    try:
        x, y = min(char_area_list, key=lambda c: c['x'])['x'], min(char_area_list, key=lambda c: c['y'])['y']
        xmax, ymax = max(char_area_list, key=lambda c: c['x']), max(char_area_list, key=lambda c: c['y'])
        w, h = xmax['x'] + xmax['w'] - x, ymax['y'] + ymax['h'] - y
        x, y, w, h = round(x, 1), round(y, 1), round(w, 1), round(h, 1)
        return x, y, w, h
    except ValueError:
        log.error('char_area_list must not empty!')
        return 0, 0, 0, 0


def get_splited_sentence(paras, mode='auto'):
    """
    划分paragraph　　按符号分割
    :param paras:
    :param mode: 'auto', 'ch', 'en'
    :return:
    """
    """将段落拆成句子并映射到段落id"""
    zh_re = re.compile(r'([;；。])')
    sentences = []
    for p in paras:
        p1, p2 = 0, 0
        round_mode = mode
        if mode == 'auto':
            round_mode = 'en' if len(re.findall(r'[A-Za-z]', p['text'])) / float(len(p['text'])) > 0.5 else 'ch'
        splited_s = zh_re.split(p['text']) if round_mode == 'ch' else sent_tokenize(p['text'])
        for s in splited_s:
            if not s or s.replace(" ", "") == "":
                continue
            if round_mode == 'ch' and zh_re.search(s):
                p1 += 1
                p2 += 1
                continue
            # 去掉句子开头的空格
            blank_start_match = re.search(r"^\s+", s)
            not_blank_start = blank_start_match.end() if blank_start_match else 0
            p1 += not_blank_start
            p2 += not_blank_start
            s = s[not_blank_start:]
            # 去掉句子末尾的空格
            blank_start_match = re.search(r"\s+$", s)
            blank_start = blank_start_match.start() - 1 if blank_start_match else len(s)
            s = s[:blank_start_match.start()] if blank_start_match else s
            p2 += blank_start
            s_char_area_list = p['char_area_list'][p1: p2 + 1]
            for dct in s_char_area_list:
                for k in dct.keys():
                    dct[k] = round(dct[k], 1)
            x, y, w, h = chars2bbox(s_char_area_list)
            # if len(s) != len(s_char_area_list):
            #     log.warning('text and char_area_list length does not match! sid: %s')
            sentences.append({
                'text': s,
                'sid': len(sentences),
                'pid': p['pid'],
                'page_index': p['page_index'],
                'char_area_list': s_char_area_list,
                'is_footer': p.get('is_footer'),
                'is_header': p.get('is_header'),
                'x': x, 'y': y, 'w': w, 'h': h,
                'font_size': p.get('font_size')
            })
            p1 += len(s) - 1
    return sentences


# if __name__ == '__main__':
#     import json
#     with open('all.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     sentences_list = []
#     paras, params = get_paragraph(data['pages'])
#     sentences = get_splited_sentence(paras)
#     for s in sentences:
#         sentences_list.append(s['text'])