import hanlp
import re

# Load pretrained models for tokenization and dependency parsing
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)

quantity_units = ["人", "个", "辆", "架", "艘", "发", "枚", "门", "套"]

# Text to be segmented and parsed
# text = [
#     '7月17日，海军第四驱逐舰支队的青岛舰（052C驱逐舰，舷号113）和昆明舰（052D驱逐舰，舷号172）共492人，在东海进行防空演习，演习期间，昆明舰发射红旗-9防空导弹2枚，成功拦截高空模拟目标。7月18日，空军第七战斗机师的歼-11B战斗机（代号J-11B）14架，在西北某靶场进行空对地打击训练，此次训练中，歼-11B战斗机成功发射6枚空对地导弹，摧毁了模拟敌方地面目标。'
# ]


def merge_quantity_phrases(tokens):
    result = []
    i = 0

    while i < len(tokens):
        if re.match(r'\d+', tokens[i]) and i + 1 < len(tokens) and tokens[i + 1] in quantity_units:
            result.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result

def extract_related_words(sentence, word):
    related_words = []
    queue = [word]
    while queue:
        current_word = queue.pop(0)
        for w in sentence:
            if w['head'] == current_word['id']:
                related_words.append(w['form'])
                queue.append(w)
    return related_words

def is_numeric_word(word):
    for char in quantity_units:
        if char in word:
            numeric_part = word.split(char)[0]
            if numeric_part.isdigit():
                return True
    return False

def extract_numeric_info(dep_result):
    numeric_info = {}
    for sentence in dep_result:
        for word in sentence:
            if is_numeric_word(word['form']):
                numeric_key = word['form']
                related_words = []
                head_id = word['head']
                for w in sentence:
                    if w['id'] == head_id:
                        related_words.append(w['form'])
                        related_words.extend(extract_related_words(sentence, w))
                    if w['deprel'] == 'nsubj':
                        related_words.append(w['form'])
                related_words = list(set(related_words))
                numeric_info[numeric_key] = related_words
    return numeric_info

# # Perform tokenization
# seg_text = tok(text)
# print("Segmentation Result:", seg_text)
#
# # Apply post-processing to the tokenized text
# post_processed_text = [merge_quantity_phrases(sentence) for sentence in seg_text]
# print("Post-processed Result:", post_processed_text)
#
# # Perform dependency parsing on the post-processed text
# dep_text = dep(post_processed_text)
#
# # Extract and print numeric information
# numeric_info = extract_numeric_info(dep_text)
# for info in numeric_info:
#     print(info)
