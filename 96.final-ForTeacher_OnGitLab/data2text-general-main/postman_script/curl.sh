#!/bin/bash

echo "before infer test"

curl -d @infer.json \
    -o infer_result.json
    -H "Content-Type: application/json" \
    -X POST http://172.20.2.202:5002

echo "before train test"

curl -d @train_data.json \
    -o train_result.json
    -H "Content-Type: application/json" \
    -X POST http://172.20.2.202:5002/train

echo "after test"