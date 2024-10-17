import json

def extract_keywords(verify, origin):
    keywords = []
    for row_key, row in verify.items():
        for col_idx, value in enumerate(row):
            if value["v"] is not None:  # Check if the "v" key is not None
                col_letter = chr(65 + col_idx)  # Convert column index to letter (A, B, C, ...)
                file_with_xlsx = origin[row_key]["file"] #+ ".xlsx"
                sheet = origin[row_key]["sheet"]
                cell = f"{file_with_xlsx}&{sheet}&{col_letter}{row_key}"
                keywords.append({"words": value["v"], "index": [cell]})
    return keywords


def get_unique_files_and_sheets(data):
    files = set()
    sheets = set()
    for entry in data:
        for row_key, origin in entry["origin"].items():
            files.add(origin["file"] + "") #.xlsx")
            sheets.add(f"{origin['file']}&{origin['sheet']}") #.xlsx")
    return "&".join(files), ",".join(sheets)

def merge_keywords(records):
    merged_keywords = []
    content = ""
    for record in records:
        content = record["content"]
        merged_keywords.extend(record["keywords"])
    return {
        "id": "",
        "content": content,
        "keywords": merged_keywords
    }

def convert_model_output_2_output(csc_data, scut_data):
    csc_file, csc_sheets = get_unique_files_and_sheets(csc_data)
    scut_file, scut_sheets = get_unique_files_and_sheets(scut_data)

    file_name = "&".join(set(csc_file.split("&")))
    sheet_name = ",".join(set(csc_sheets.split(",")))

    csc_records = []

    for csc in csc_data:
        csc_keywords = extract_keywords(csc["verify"], csc["origin"])
        csc_records.append({
            "content": csc["text"],
            "keywords": csc_keywords
        })


    scut_records = []

    for scut in scut_data:
        scut_keywords = extract_keywords(scut["verify"], scut["origin"])
        scut_records.append({
            "content": scut["text"],
            "keywords": scut_keywords
        })


    merged_csc_record = merge_keywords(csc_records)
    merged_scut_record = merge_keywords(scut_records)

    return {
        "code": 0,
        "message": "success",
        "data": [
            {
                "fileName": file_name,
                "sheetsData": [
                    {
                        "sheet": sheet_name,
                        "records": [{
                            "csc": merged_csc_record,
                            "scut": merged_scut_record
                        }]
                    }
                ]
            }
        ]
    }
