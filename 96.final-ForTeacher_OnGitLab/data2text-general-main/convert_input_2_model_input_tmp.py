import json
import re

def convert_input_2_model_input(data):
    def parse_origin_json(data):
        tables = []
        table_map = {}
        text = ""
        for whole_data_item in data['wholeData']:
            for excel_data_item in whole_data_item['excelData']:
                text = excel_data_item['text']  # 如果过长则需要报错
                segments = excel_data_item['segment']
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
                "origin": content["origin"]
            })
        return tables, text

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

    def write_to_model_json(tables, text):
        final_data = []
        for table in tables:
            final_data.append({
                "table": table["table"],
                "origin": table["origin"],
                "text": text
            })
        return final_data

    tables, text = parse_origin_json(data)
    
    return write_to_model_json(tables, text)