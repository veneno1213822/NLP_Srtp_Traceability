# import hanlp
import re
from fuzzywuzzy import fuzz
from word2number import w2n
import itertools

# 中文数字对应的阿拉伯数字
chinese_to_arabic = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000
}

# Load pretrained models for tokenization and dependency parsing
# tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)

# 预定义量词库
area_headers = {"地点"}
quantity_units = {"人", "个", "辆", "架", "艘", "发", "枚", "门", "套"}
target_headers = {"人数", "数量", "装备数量"}
equipment_headers = {"武器", "装备", "装备名称"}

# 将中文数字转换为阿拉伯数字
def chinese_number_to_arabic(chinese_num):
    result = 0
    temp_num = 0
    unit = 1
    for char in reversed(chinese_num):
        if char in chinese_to_arabic:
            digit = chinese_to_arabic[char]
            if digit >= 10:
                if digit > unit:
                    unit = digit
                else:
                    unit *= digit
            else:
                temp_num += digit * unit
        if char in "万亿":
            result += temp_num
            temp_num = 0
            unit = chinese_to_arabic[char]

    result += temp_num
    return result


# 将英文数字转换为阿拉伯数字
def english_number_to_arabic(english_num):
    try:
        return w2n.word_to_num(english_num)
    except ValueError:
        return english_num  # 如果无法转换，保持原样


def preprocess_chinese_and_english_numbers(text):
    # 定义匹配中文数字的正则表达式模式
    chinese_pattern = re.compile(r'[零一二两三四五六七八九十百千万亿]+')

    # 定义匹配英文数字的正则表达式模式
    english_pattern = re.compile(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)+[\s|-]*(and[\s|-]*)?(\d+)?\b',
        re.IGNORECASE)

    # 替换中文数字为阿拉伯数字
    def replace_chinese_match(match):
        chinese_num = match.group(0)
        arabic_num = chinese_number_to_arabic(chinese_num)
        return str(arabic_num)

    # 替换英文数字为阿拉伯数字
    def replace_english_match(match):
        english_num = match.group(0)
        arabic_num = english_number_to_arabic(english_num)
        return str(arabic_num)

    # 处理中文数字
    processed_text = re.sub(chinese_pattern, replace_chinese_match, text)

    # 处理英文数字
    processed_text = re.sub(english_pattern, replace_english_match, processed_text)

    return processed_text

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

# 分词并依存文法分析
# def depandency_analysis(data):
#     text = data[0]['text']
#     text = [preprocess_chinese_and_english_numbers(text)]
#     # Load pretrained models for tokenization and dependency parsing
#     tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
#     dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
#     seg_text = tok(text)
#     post_processed_text = [merge_quantity_phrases(sentence) for sentence in seg_text]
#     dep_text = dep(post_processed_text)
#     numeric_info = extract_numeric_info(dep_text)
#     print(numeric_info)
#     return numeric_info

def find_most_similar_key(dict_input, str_input):
    max_similarity = 0
    most_similar_key = None

    # 遍历字典中的每个键值对
    for key, value_list in dict_input.items():
        # 遍历每个值，计算与输入字符串的相似度
        for value in value_list:
            similarity = fuzz.ratio(str_input, value)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key

    return most_similar_key

def extract_quantity_phrases_with_spans(text, quantity_units, base_offset=0):
    pattern = r'(\d+\s*(' + '|'.join(re.escape(unit) for unit in quantity_units) + '))'
    matches = re.finditer(pattern, text)
    return [(match.group(1), match.start() + base_offset, match.end() + base_offset) for match in matches]

def get_index(headers, target_values):
    for index, header in enumerate(headers):
        if header in target_values:
            return index
    return None


def find_combinations_that_sum_to_target(candidates, target):
    valid_combinations = []

    # 遍历所有可能的数字组合
    for r in range(1, len(candidates) + 1):
        for combination in itertools.combinations(candidates, r):
            if sum(combination) == target:
                valid_combinations.append(combination)

    return valid_combinations


def add_refs_new(data):
    # numeric_info = depandency_analysis(data)

    # 处理每个对象
    for item in data:
        add_refs_dict = {}
        text = item['text']
        text = preprocess_chinese_and_english_numbers(text)

        # 获取header和索引
        headers = item['table']['header']
        area_index = get_index(headers, area_headers)
        count_index = get_index(headers, target_headers)
        equipment_index = get_index(headers, equipment_headers)

        if area_index is None or count_index is None or equipment_index is None:
            continue  # 如果没有找到“地点”或“人数”/“数量”/“装备数量”，跳过该项

        # 拆分文本为子句并记录每个子句的起始位置
        sentences = re.split(r'(?<=[。｡])', text)
        sentence_start_positions = []
        start_pos = 0
        for sentence in sentences:
            sentence_start_positions.append(start_pos)
            start_pos += len(sentence)

        for sentence, base_offset in zip(sentences, sentence_start_positions):
            # 找到句子中的数量短语
            quantity_phrases_with_spans = extract_quantity_phrases_with_spans(sentence, quantity_units, base_offset)

            # 获取文本中的目标数字，如"八个人"中的“8”
            for quantity_phrase, start, end in quantity_phrases_with_spans:
                numeric_value = int(re.findall(r'\d+', quantity_phrase)[0])  # 提取数量短语中的数字

                # 找到表格中符合"地点"的行
                candidate_rows = []
                for row_key, row in item['table']['data'].items():
                    if row[area_index] in sentence:  # 如果"地点"匹配
                        count_value = row[count_index].strip()  # 获取数量字段的值并去除首尾空白
                        # 使用正则表达式提取出数字部分
                        match = re.match(r'\d+', count_value)
                        if match:
                            count_num = match.group(0)  # 提取数字部分
                            candidate_rows.append((row_key, int(count_num)))  # 将数字转换为整数并添加到候选行
                        else:
                            continue  # 如果没有数字部分，跳过该行

                # 获取可能相加等于numeric_value的组合
                candidate_values = [val[1] for val in candidate_rows]  # 取出人数
                valid_combinations = find_combinations_that_sum_to_target(candidate_values, numeric_value)

                # 如果有合法的组合
                if valid_combinations:
                    for combination in valid_combinations:
                        for row_key, row_value in candidate_rows:
                            if row_value in combination:
                                if quantity_phrase not in add_refs_dict:
                                    add_refs_dict[quantity_phrase] = {
                                        "text": quantity_phrase,
                                        "span": [start, end],
                                        "ref": {}
                                    }
                                if row_key not in add_refs_dict[quantity_phrase]["ref"]:
                                    add_refs_dict[quantity_phrase]["ref"][row_key] = {}
                                add_refs_dict[quantity_phrase]["ref"][row_key][headers[count_index]] = str(row_value)

        # 将字典转换为列表
        add_refs = []
        for quantity_phrase in add_refs_dict:
            add_refs.append(add_refs_dict[quantity_phrase])

        # 添加“add_refs”字段
        item['add_refs'] = add_refs

    return data
