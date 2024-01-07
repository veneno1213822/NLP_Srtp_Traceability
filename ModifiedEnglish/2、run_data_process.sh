#source ~/.bashrc
#conda activate Python37withTensorFlow1155
electra_path="/data1/srtp001/tmp-Nlp_srtp/Python_Project-001-NLP_SRTP/96.2-ModifiedEnglish_2021ReadingComprehensionModel/baseline_English/models/bert-base-uncased" #electra_base" #绝对路径。相对路径会因为bash运行默认~/.bashrc，找不到路径
python data_process.py \
          --input_file data/Prepared_train_data.json \
          --for_training \
          --output_prefix train \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/

python data_process.py \
          --input_file data/Prepared_dev_data.json \
          --output_prefix dev \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/