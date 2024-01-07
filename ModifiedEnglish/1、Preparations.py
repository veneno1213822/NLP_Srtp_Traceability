import json
import logging
from tqdm import tqdm

import torch
import argparse



def ExtractInfomation_FromTottoJson(file_path, maxNum):
    final_sentences_forIndex = []
    # 以下字典均以final_sentence为key
    context_dic = {}
    question_dic = {}
    answer_start_dic = {}
    answer_text_dic = {}

    # 打开.jsonl文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for i, line in tqdm(enumerate(file)):  # 一行只有一个final_sentence，拿它来当字典索引
            if i < maxNum:
                # 将原始数据解析为JSON格式
                data = json.loads(line)
                # 提取所有"value"部分
                values = [cell["value"] for row in data["table"] for cell in row]
                # 将"value"部分拼成一个文段，得到context
                context = "，".join([f'“{value}”' for value in values])#<class 'str'>
                final_sentence = data["sentence_annotations"][0]["final_sentence"]
                question = f'What contents in the table form this sentence "{final_sentence}" ?'
                highlighted_cells_positions = data["highlighted_cells"]

                answer_starts = []
                high_lighted_texts = []
                for position in highlighted_cells_positions:
                    row_index_high = position[0]
                    col_index_high = position[1]
                    for row_index, row in enumerate(data["table"]):
                        if row_index == row_index_high:
                            for col_index, cell in enumerate(row):
                                if col_index == col_index_high:
                                    high_lighted_text = data["table"][row_index][col_index]
                                    high_lighted_text = high_lighted_text['value']
                                    answer_start = context.find(high_lighted_text)
                                    answer_starts.append(answer_start)
                                    high_lighted_texts.append(high_lighted_text)
                answer_start_dic[final_sentence] = answer_starts
                answer_text_dic[final_sentence] = high_lighted_texts

                context_dic[final_sentence] = context
                question_dic[final_sentence] = question
                final_sentences_forIndex.append(final_sentence)
            else:
                break

    # print('context_dic:', context_dic)
    # print('question_dic', question_dic)
    # print('answer_start_dic:', answer_start_dic)
    # print('answer_text_dic', answer_text_dic)
    # print('final_sentences_forIndex:', final_sentences_forIndex)
    return context_dic, question_dic, answer_start_dic, answer_text_dic, final_sentences_forIndex

def WriteInformation_toJSON(Contexts_FromValues, Questions, AnswerStarts, AnswerTexts, FinalSentenceForIndex, OutputJsonPath):
    data_list = []
    for i, final_sentence in enumerate(FinalSentenceForIndex):
        if len(AnswerStarts[final_sentence]) == len(AnswerTexts[final_sentence]):
            starts = AnswerStarts[final_sentence]
            # print(type(starts))#<class 'list'>[1, 99]
            # input()
            texts = AnswerTexts[final_sentence]

            context = Contexts_FromValues[final_sentence]
            casename = "Text to Table Traceability"
            qas_list = [
                {
                    "question": Questions[final_sentence],
                    "answers": [{"answer_start": starts[i], "text": texts[i]} for i in range(len(starts))],
                    "id": i,
                    "is_impossible": "false"
                }
            ]
            paragraphs_dict = {"context": context, "casename": casename, "qas": qas_list}
            data_list.append({"paragraphs": [paragraphs_dict], "caseid": f"CaseID {i}"})

        else:
            print('Error !!!!!')
            return 0

    # 创建最终的JSON结构
    json_data = {"data": data_list, "version": "1.0"}
    # 将JSON数据转换为字符串
    json_string = json.dumps(json_data, indent=2)  # indent参数用于格式化输出，可选

    # 打印或保存到文件
    # print(json_string)
    # 如果需要保存到文件，可以使用以下代码
    with open(OutputJsonPath, "w") as json_file:
        json_file.write(json_string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Totto_TrainJson_path",
        default='./data/totto_train_data.jsonl',#totto_train_data.jsonl TottoData_table.jsonl',
        type=str,
        # required=True,
        help="Path to original Totto train json."
    )
    parser.add_argument(
        "--Output_TrainJson_path",
        default='./data/Prepared_train_data.json',
        type=str,
        # required=True,
        help="Path to after-preparation train json."
    )
    parser.add_argument(
        "--Totto_DevJson_path",
        default='./data/totto_dev_data.jsonl',#totto_dev_data.jsonl TottoData_table.jsonl',
        type=str,
        # required=True,
        help="Path to original Totto dev json."
    )
    parser.add_argument(
        "--Output_DevJson_path",
        default='./data/Prepared_dev_data.json',
        type=str,
        # required=True,
        help="Path to after-preparation dev json."
    )
    parser.add_argument(
        "--Train_NumToPrepare",
        default=43066,
        type=int,
        # required=True,
        help="只训练这么多数量的训练集"
    )
    parser.add_argument(
        "--Dev_NumToPrepare",
        default=2054,
        type=int,
        # required=True,
        help="只训练这么多数量的验证集"
    )
    args = parser.parse_args()
    logging.info("All input parameters:")
    print(json.dumps(vars(args), sort_keys=False, indent=2))

    print("*********Train Set Preparation Begin:*********")
    trainContexts_FromValues, train_Questions, train_AnswerStarts, train_AnswerTexts, train_FinalSentenceForIndex = \
        ExtractInfomation_FromTottoJson(args.Totto_TrainJson_path, args.Train_NumToPrepare)
    WriteInformation_toJSON(trainContexts_FromValues, train_Questions, train_AnswerStarts, train_AnswerTexts,
                            train_FinalSentenceForIndex, args.Output_TrainJson_path)
    print("*********Train Set Preparation Done.*********\n")

    print("*********Development Set Preparation Begin:*********")
    devContexts_FromValues, dev_Questions, dev_AnswerStarts, dev_AnswerTexts, dev_FinalSentenceForIndex = \
        ExtractInfomation_FromTottoJson(args.Totto_DevJson_path, args.Dev_NumToPrepare)
    WriteInformation_toJSON(devContexts_FromValues, dev_Questions, dev_AnswerStarts, dev_AnswerTexts,
                            dev_FinalSentenceForIndex, args.Output_DevJson_path)
    print("*********Development Set Preparation Done.*********\n")


if __name__ == "__main__":
    torch.cuda.set_device(0)  # 0表示第一个GPU，1表示第二个GPU，以此类推
    main()
    print("Preparation Done.")