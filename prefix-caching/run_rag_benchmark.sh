#!/bin/bash

DATASET="./sharegpt_real_rag_prefix.json"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "========================================="
echo "开始工业级 RAG Prefix Caching 压测"
echo "========================================="

# 使用泊松分布模拟真实流量，平均每秒 5 个请求 (qps=5)，持续发送 100 个请求
# 这种流量模型比 inf (瞬间并发) 更能反映真实生产环境的延迟表现
vllm bench serve \
    --model $MODEL \
    --backend vllm \
    --dataset sharegpt \
    --dataset-path $DATASET \
    --num-prompts 100 \
    --request-rate poisson \
    --poisson-request-interval 0.2 \
    --port 8000 \
    --save-result

echo "压测完成！请查看生成的 result_*.json 文件。"