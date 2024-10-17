import json
import re

def convert_input_2_model_input(data):
    def parse_origin_json(excel_data):
        tables = []
        table_map = {}
        text = excel_data['text']  # 如果过长则需要报错
        segments = excel_data['segment']
        for segment in segments:
            header = tuple(
                cell["columnName"] for cell in segment if isinstance(cell, dict) and cell["columnName"] != "id")
            if header not in table_map:
                table_map[header] = {
                    "header": list(header),
                    "data": {},
                    "origin": {}
                }
            for cell in segment:
                if isinstance(cell, dict):
                    row_num = extract_row_num(cell["position"])
                    if row_num is not None and row_num not in table_map[header]["data"]:
                        table_map[header]["data"][row_num] = []
                    if cell["columnName"] != "id" and row_num is not None:
                        table_map[header]["data"][row_num].append(cell["columnValue"])
                    if row_num is not None and row_num not in table_map[header]["origin"]:
                        table_map[header]["origin"][row_num] = {
                            "file": cell["position"].split('&')[0],
                            "sheet": cell["position"].split('&')[1],
                            "id": ""
                        }
                    if cell["columnName"] == "id" and row_num is not None:
                        table_map[header]["origin"][row_num]["id"] = cell["columnValue"]

        for header, content in table_map.items():
            tables.append({
                "table": {
                    "header": content["header"],
                    "data": content["data"]
                },
                "origin": content["origin"],
                "text": text  # 保留每个 excelData 的文本
            })
        return tables

    def extract_row_num(position):
        # 先判断是否为[12,0]这种格式
        if '[' in position and ']' in position:
            try:
                # 提取括号内的内容并解析为列表，然后取第一个元素作为行号
                row_num = eval(position.split('&')[-1])[0]
            except:
                row_num = None
        else:
            # 处理A13这种格式，使用正则表达式提取数字部分
            match = re.search(r'\d+', position.split('&')[-1])
            row_num = int(match.group(0)) if match else None
        return row_num

    def write_to_model_json(whole_data_results):
        final_data = []
        for tables in whole_data_results:
            final_data.append(tables)
        return final_data

    # 处理多个 wholeData
    whole_data_results = []
    for whole_data_item in data['wholeData']:
        # 对每个 wholeData 中的多个 excelData 进行处理
        temp_tables = []
        for excel_data_item in whole_data_item['excelData']:
            temp_tables.append(parse_origin_json(excel_data_item))
        whole_data_results.extend(temp_tables)

    # 返回处理结果，去掉最外层的多余列表
    return whole_data_results
