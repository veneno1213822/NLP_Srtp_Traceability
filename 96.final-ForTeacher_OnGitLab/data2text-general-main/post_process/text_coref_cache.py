from fuzzywuzzy import fuzz
import ngram

# 用于缓存计算结果，减少重复计算
def cache_results(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper


# 用装饰器缓存fuzz.ratio和ngram_similarity结果
@cache_results
def cached_fuzz_ratio(word, substring):
    return fuzz.ratio(word, substring)


@cache_results
def cached_ngram_similarity(word, substring, n=3):
    ngram_index = ngram.NGram(N=n)
    return ngram_index.compare(word, substring)


def is_subsequence(word, string):
    # 初始化 word 和 string 的索引
    word_index = 0
    string_index = 0
    # 遍历 string，检查是否能按照顺序匹配 word 中的字符
    while word_index < len(word) and string_index < len(string):
        if word[word_index] == string[string_index]:
            word_index += 1
        string_index += 1
    # 如果 word_index 达到了 word 的长度，说明所有字符都匹配了
    return word_index == len(word)

def check_substrings(word, substrings_tmp):
    for substring in substrings_tmp:
        if is_subsequence(word, substring):
            return True, substring
    return False, ""

def compute_cf_fuzz(similarity, threshold_fuz):
    step_1 = int(threshold_fuz + (100 - threshold_fuz) * 2 / 3)  # 76
    step_2 = int(threshold_fuz + (100 - threshold_fuz) / 3)  # 53
    if step_1 < similarity <= 100:
        confidence = 100
    elif step_2 < similarity <= step_1:
        confidence = 95
    elif threshold_fuz < similarity <= step_2:
        confidence = 90
    elif threshold_fuz / 2 < similarity <= threshold_fuz:
        confidence = round(((similarity / 100) ** 2) * 100)  # 平方
    elif 0 <= similarity <= threshold_fuz / 2:
        confidence = 0
    return confidence


def compute_cf_ngram(similarity, threshold_ngram):
    step_1 = threshold_ngram + (1 - threshold_ngram) * 2 / 3  # 0.6867
    step_2 = threshold_ngram + (1 - threshold_ngram) / 3  # 0.3733
    if step_1 < similarity <= 1:
        confidence = 100
    elif step_2 < similarity <= step_1:
        confidence = 95
    elif threshold_ngram < similarity <= step_2:
        confidence = 90
    elif threshold_ngram / 2 < similarity <= threshold_ngram:
        confidence = round((similarity ** 2) * 100)  # 平方
    elif 0 <= similarity <= threshold_ngram / 2:
        confidence = 0
    return confidence


def extract_substrings(sentence, min_len, max_len):
    substrings = []
    length = len(sentence)
    for window_len in range(min_len, max_len + 1):
        for start in range(length - window_len + 1):
            substrings.append(sentence[start:start + window_len])
    return substrings


def extract_sub_needs_1(tmp_substrings, word):
    filtered_substrings = []
    for substring in tmp_substrings:
        # 检查 简称substring首位 是否与 全称word首位 相同
        if word[0] == substring[0]:
            filtered_substrings.append(substring)
    return filtered_substrings


def extract_sub_needs_2(tmp_substrings, word):  # 去掉与word一个字都不相同的substring(普通溯源结果肯定得有相同字)
    filtered_substrings = []
    for substring in tmp_substrings:
        # 检查 substring 是否与 word 有任何相同的字符
        if any(char in word for char in substring):
            filtered_substrings.append(substring)
    return filtered_substrings


def Extract_FromInputData(input_data):
    wholeData = input_data[0]
    return wholeData["table"], wholeData["origin"], wholeData["text"], wholeData["verify"], wholeData["add_refs"]


def CompleteVerify(table, origin, text, verify, add_refs, model_path, threshold_fuz, threshold_ngram):
    sentence = text
    verify_new = verify
    MeaninglessWords = ["不详", "无特定", "未明确", "未提及", "未提供"]
    punctuations = ["，", "、", "；", "。", ";", ",", "､", "｡"]

    # model = SentenceTransformer(model_path)  # 支持多语言的预训练模型，放在循环外面以避免重复加载
    for key_verify, list_verify in verify.items():
        list_verify_new = list_verify
        for i_inList, answer in enumerate(list_verify):
            answer_v = str(answer["v"])  # JSON里"v": null会让answer["v"]=None,str(answer["v"])="None"
            if answer_v == "None" or answer_v == "":  # 原来没识别到答案：补充 全称简称、模型遗漏 等
                word = str(table["data"][key_verify][i_inList])
                if word != "None" and word != "" and any(w_meaningless in word for w_meaningless in MeaninglessWords) and not any(
                        p_meaningless in word for p_meaningless in punctuations):
                    pass  # 如果原表格这里就是“未明确”这种无信息的，则什么都不做
                else:  # 原表格这里是正常信息或者是空的：补充（空的肯定过不了阈值，与text无关的表格也过不了阈值）
                    list_verify_new = integrate_multiple_methods(list_verify_new, i_inList, word, sentence,
                                                                 threshold_fuz, threshold_ngram, 'None')
            else:  # 原来识别到答案：补充细节
                word = answer_v
                list_verify_new = integrate_multiple_methods(list_verify_new, i_inList, word, sentence, threshold_fuz,
                                                             threshold_ngram, 'Ori')
        verify_new[key_verify] = list_verify_new
    return table, origin, text, verify_new, add_refs


def integrate_multiple_methods(list_verify_new, i_inList, word, sentence, threshold_fuz, threshold_ngram, NoneOrOri):
    if NoneOrOri == 'None':  # 这是原来没有，准备新补充的全称简称。
        if len(word) > 5:  # word是全称
            substrings_tmp = extract_substrings(sentence, min_len=2, max_len=5)
            substrings = extract_sub_needs_1(substrings_tmp, word)  # 去掉 首位 与 全称word首位 不相同的 简称substring
            similarity_max_fuz, similarity_max_ngram = -1, -1
            similar_word_fuz, similar_word_ngram = "", ""
            for substring in substrings:
                similarity_fuz = cached_fuzz_ratio(word, substring)
                similarity_ngram = cached_ngram_similarity(word, substring)
                if similarity_fuz > similarity_max_fuz:
                    similarity_max_fuz = similarity_fuz
                    similar_word_fuz = substring
                if similarity_ngram > similarity_max_ngram:
                    similarity_max_ngram = similarity_ngram
                    similar_word_ngram = substring
            is_abbreviation_fuz = similarity_max_fuz > threshold_fuz
            is_abbreviation_ngram = similarity_max_ngram > threshold_ngram
            if is_abbreviation_fuz and is_abbreviation_ngram:
                if len(similar_word_fuz) > len(similar_word_ngram):
                    similar_word = similar_word_fuz
                    confidence = compute_cf_fuzz(similarity_max_fuz, threshold_fuz)
                else:
                    similar_word = similar_word_ngram
                    confidence = compute_cf_ngram(similarity_max_ngram, threshold_ngram)
                start_index = sentence.find(similar_word)
                end_index = start_index + len(similar_word) - 1
                span = [start_index, end_index]
                list_verify_new[i_inList] = {"v": word, "span": span, "cf": confidence}
        elif 2 <= len(word) <= 5:  # word是简称
            substrings_tmp = extract_substrings(sentence, min_len=len(word) + 1, max_len=10)
            if_find, similar_word = check_substrings(word, substrings_tmp)
            if if_find:
                start_index = sentence.find(similar_word)
                end_index = start_index + len(similar_word) - 1
                span = [start_index, end_index]
                list_verify_new[i_inList] = {"v": word, "span": span, "cf": 95}
    elif NoneOrOri == 'Ori':  # 这是原来就有的
        try:
            def process_word(sentence, word, i_inList, min_len, max_len, threshold_fuz, threshold_ngram):
                substrings_tmp = extract_substrings(sentence, min_len=min_len, max_len=max_len)
                substrings = extract_sub_needs_2(substrings_tmp, word)  # 去掉与word一个字都不相同的substring
                similarity_max_fuz, similarity_max_ngram = -1, -1
                similar_word_fuz, similar_word_ngram = "", ""

                for substring in substrings:
                    similarity_fuz = cached_fuzz_ratio(word, substring)
                    similarity_ngram = cached_ngram_similarity(word, substring)
                    if similarity_fuz > similarity_max_fuz:
                        similarity_max_fuz = similarity_fuz
                        similar_word_fuz = substring
                    if similarity_ngram > similarity_max_ngram:
                        similarity_max_ngram = similarity_ngram
                        similar_word_ngram = substring

                is_abbreviation_fuz = similarity_max_fuz > threshold_fuz
                is_abbreviation_ngram = similarity_max_ngram > threshold_ngram

                if is_abbreviation_fuz and is_abbreviation_ngram:
                    if len(similar_word_fuz) > len(similar_word_ngram):
                        similar_word = similar_word_fuz
                        confidence = compute_cf_fuzz(similarity_max_fuz, threshold_fuz)
                    else:
                        similar_word = similar_word_ngram
                        confidence = compute_cf_ngram(similarity_max_ngram, threshold_ngram)

                    start_index = sentence.find(similar_word)
                    end_index = start_index + len(similar_word) - 1
                    span = [start_index, end_index]
                    list_verify_new[i_inList] = {"v": word, "span": span, "cf": confidence}

            if len(word) <= 4:
                process_word(sentence, word, i_inList, min_len=2, max_len=4, threshold_fuz=threshold_fuz,
                                 threshold_ngram=threshold_ngram)
            elif 4 < len(word) <= 8:
                process_word(sentence, word, i_inList, min_len=4, max_len=7, threshold_fuz=threshold_fuz,
                                 threshold_ngram=threshold_ngram)
            elif 8 < len(word) <= 12:
                process_word(sentence, word, i_inList, min_len=8, max_len=10, threshold_fuz=threshold_fuz,
                                 threshold_ngram=threshold_ngram)
            elif 12 < len(word) <= 15:
                process_word(sentence, word, i_inList, min_len=10, max_len=13, threshold_fuz=threshold_fuz,
                                 threshold_ngram=threshold_ngram)
            elif len(word) > 15:
                process_word(sentence, word, i_inList, min_len=13, max_len=15, threshold_fuz=threshold_fuz,
                                 threshold_ngram=threshold_ngram)

        except:  # 模型得到了与text无任何字相同的答案
            print(f"模型得到了与text无任何字相同的答案：word == {word}, sentence == {sentence}")
    return list_verify_new


def Prepare_OutputData(table, origin, text, verify_new, add_refs):
    wholeOutput = []
    whole_dic = {
        "table": table,
        "origin": origin,
        "text": text,
        "verify": verify_new,
        "add_refs": add_refs
    }
    wholeOutput.append(whole_dic)
    return wholeOutput


def coref(input_data):
    model_path = ""
    threshold_fuz, threshold_ngram = 30, 0.06
    table, origin, text, verify, add_refs = Extract_FromInputData(input_data)
    table, origin, text, verify_comple, add_refs = CompleteVerify(table, origin, text, verify, add_refs, model_path,
                                                                  threshold_fuz, threshold_ngram)
    output_data = Prepare_OutputData(table, origin, text, verify_comple, add_refs)
    return output_data
