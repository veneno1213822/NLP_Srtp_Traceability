import json
import argparse
from fuzzywuzzy import fuzz


check_header = ["任务状态", "备注", "任务", "行动", "事件"]

# 将OneKE模型的结果直接添加到input中，并生成新的output（多了verify字段，verify来自匹配成功的result）
def add_verify_data(input_data, result_data):
    data = input_data
    result_data = result_data  # 直接使用传入的result_data

    # 遍历每个元素
    for item in data:
        headers = item['table']['header']
        table_data = item['table']['data']
        verify = {}

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
                    verify[row_key].append(None)
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
                    verify[row_key].append(verify_value)
                else:
                    verify[row_key].append(None)

        item['verify'] = verify

    return data


# 对text分句
def split_text(text):
    # 按逗号和句号分句
    sentences = text.replace('，', ',').replace('。', '.').split(',')
    sub_sentences = []
    for sentence in sentences:
        sub_sentences.extend(sentence.split('.'))
    return [s.strip() for s in sub_sentences if s.strip()]


# 滑动窗口
def sliding_window_compare(value, sub_sentences):
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

        # 遍历table_data中的每一行
        for row_key, row_values in table_data.items():
            for index, value in enumerate(row_values):
                header = headers[index]  # 获取当前值对应的header

                # 判断条件：字符串长度大于等于7，或header为“任务状态”或“备注”
                if len(value) >= 7 or header in check_header:
                    best_match = sliding_window_compare(value, sub_sentences)

                    if best_match is not None:
                        # 确保verify字典中有对应的key
                        if row_key not in verify:
                            verify[row_key] = [None] * len(row_values)
                        # 将匹配的值放到verify中相应的位置
                        verify[row_key][index] = value

        # 更新item的verify字段
        item['verify'] = verify

    return data
