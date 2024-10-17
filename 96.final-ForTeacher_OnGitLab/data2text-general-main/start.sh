#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
flask --app http_api.py run --host 0.0.0.0