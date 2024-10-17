import json
import argparse
import re
from fuzzywuzzy import fuzz
from word2number import w2n

check_header = ["任务状态", "备注", "任务", "行动", "事件"]
num_header = ["人数", "数量", "装备数量"]
# 中文数字对应的阿拉伯数字
chinese_to_arabic = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000
}

# 将OneKE模型的结果直接添加到input中，并生成新的output（多了verify字段，verify来自匹配成功的result）
def add_verify_data(input_data, result_data):
    data = input_data
    result_data = result_data  # 直接使用传入的result_data

    # 遍历每个元素
    for item in data:
        headers = item['table']['header']
        table_data = item['table']['data']
        verify = {}
        add_refs = []  # 初始化 add_refs 列表

        # 初始化verify字典的每一行
        for row_key in table_data.keys():
            verify[row_key] = []

        # 对每个header进行处理
        for index, header in enumerate(headers):
            # 获取对应的result_values
            result_values = result_data.get(header, [])

            # 遍历table_data中的每一行
            for row_key, row_values in table_data.items():
                value = row_values[index]

                if value == "":
                    verify[row_key].append({"v": None, "span": "", "cf": ""})
                    continue

                matched_results = []

                # 如果包含分号，进行拆分
                sub_values = value.split('；') if '；' in value else [value]

                # 与result_values中的值比较
                for sub_value in sub_values:
                    for result_value in result_values:
                        # 计算相似度
                        similarity = fuzz.ratio(sub_value.strip(), result_value)
                        if similarity > 80:
                            matched_results.append(result_value)

                if matched_results:
                    # 将多个匹配的result_value用分号连接起来
                    verify_value = '；'.join(matched_results)
                    verify[row_key].append({"v": verify_value, "span": "", "cf": ""})
                else:
                    verify[row_key].append({"v": None, "span": "", "cf": ""})

        item['verify'] = verify
        item['add_refs'] = add_refs  # 添加 add_refs 字段

    return data


# 对text分句
def split_text(text):
    # 按逗号和句号分句
    sentences = text.replace('，', ',').replace('。', '.').split(',')
    sub_sentences = []
    for sentence in sentences:
        sub_sentences.extend(sentence.split('.'))
    return [s.strip() for s in sub_sentences if s.strip()]

# 滑动窗口整句
def sliding_window_compare(value, text):
    first_char = value[0]
    best_match = None
    highest_similarity = 0

    # 找到包含首字符的位置
    start_positions = [i for i in range(len(text)) if text[i] == first_char]

    for start in start_positions:
        # 滑动窗口进行比较
        for end in range(start + 1, len(text) + 1):
            window = text[start:end]
            similarity = fuzz.ratio(value, window)
            if similarity > 50 and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = window

    return best_match

# 滑动窗口子句
def sliding_window_compare_split(value, sub_sentences):
    first_char = value[0]
    best_match = None
    highest_similarity = 0

    # 遍历每个子句
    for sub_sentence in sub_sentences:
        # 找到包含首字符的位置
        start_positions = [i for i in range(len(sub_sentence)) if sub_sentence[i] == first_char]

        for start in start_positions:
            # 滑动窗口进行比较
            for end in range(start + 1, len(sub_sentence) + 1):
                window = sub_sentence[start:end]
                similarity = fuzz.ratio(value, window)
                if similarity > 80 and similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = window

    return best_match


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

def sliding_window_compare_num(value, text):
    processed_text = preprocess_chinese_and_english_numbers(text)
    first_char = value[0]
    best_match = None
    highest_similarity = 0

    # 找到包含首字符的位置
    start_positions = [i for i in range(len(processed_text)) if processed_text[i] == first_char]

    for start in start_positions:
        # 滑动窗口进行比较
        for end in range(start + 1, len(processed_text) + 1):
            window = processed_text[start:end]
            similarity = fuzz.ratio(value, window)
            if similarity > 90 and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = window

    return best_match

# 更新verify字段：verify来自匹配成功的data
def update_verify_based_on_text(data):
    text = data[0]['text']

    # 分句
    sub_sentences = split_text(text)

    # 遍历每个元素
    for item in data:
        table_data = item['table']['data']
        headers = item['table']['header']  # 获取header
        verify = item.get('verify', {})
        add_refs = item.get('add_refs', [])  # 获取add_refs，如果不存在则初始化为空列表

        # 遍历table_data中的每一行
        for row_key, row_values in table_data.items():
            for index, value in enumerate(row_values):
                header = headers[index]  # 获取当前值对应的header

                # 判断条件：字符串长度大于等于7，或header为“任务状态”或“备注”等特定列
                if len(value) >= 7 or (header in check_header and len(value) > 0):
                    # 检查 value 中是否存在逗号或句号
                    if '，' in value or '。' in value or ',' in value:
                        best_match = sliding_window_compare(value, text)
                    else:
                        best_match = sliding_window_compare_split(value, sub_sentences)

                    if best_match is not None:
                        # 确保verify字典中有对应的key
                        if row_key not in verify:
                            verify[row_key] = [{"v": None, "span": "", "cf": ""} for _ in row_values]
                        # 将匹配的值放到verify中相应的位置
                        verify[row_key][index] = {"v": value, "span": "", "cf": ""}

                # 判断条件：header为涉及数字的信息列，且字符串长度大于0
                if header in num_header and len(value) > 0:
                    best_match = sliding_window_compare_num(value, text)

                    if best_match is not None:
                        # 确保verify字典中有对应的key
                        if row_key not in verify:
                            verify[row_key] = [{"v": None, "span": "", "cf": ""} for _ in row_values]
                        # 将匹配的值放到verify中相应的位置
                        verify[row_key][index] = {"v": value, "span": "", "cf": ""}

        # 更新item的verify字段
        item['verify'] = verify
        item['add_refs'] = add_refs  # 确保 add_refs 字段存在

    return data
