#source ~/.bashrc
#conda activate Python37withTensorFlow1155
electra_path="/data1/srtp001/tmp-Nlp_srtp/Python_Project-001-NLP_SRTP/96.1-OfficialChinese_2021ReadingComprehensionModel/baseline_Chinese/models/bert-base-chinese" #注释chinese_electra_small_L-12_H-256_A-4"绝对路径。electra_180g_large" #模型下载地址https://github.com/ymcui/Chinese-ELECTRA
python data_process.py \
          --input_file data/train.json \
          --for_training \
          --output_prefix train \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/

python data_process.py \
          --input_file data/dev.json \
          --output_prefix dev \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/