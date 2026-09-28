import os
import json
import random
from datasets import load_dataset

# 设置镜像，确保下载成功
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("1. 正在下载真实 RAG 数据集 (m-ric/huggingface_doc_qa)...")
# 加载数据集
dataset = load_dataset("m-ric/huggingface_doc_qa", split="test")

# 2. 寻找一个足够长的真实文档作为共享的 System Prompt (模拟核心知识库)
# 我们寻找长度在 1500 - 3000 字符之间的文档，确保 Prefill 阶段有足够计算量
long_contexts = [item['context'] for item in dataset if 1500 < len(item['context']) < 3000]

if not long_contexts:
    # 如果没找到合适的，就取最长的一个
    long_contexts = sorted([item['context'] for item in dataset], key=len, reverse=True)[:1]

# 选取一个最具代表性的真实文档
shared_system_prompt = long_contexts[0]
print(f"   ✅ 选定共享 System Prompt，长度: {len(shared_system_prompt)} 字符")

# 3. 抽取 100 个真实的 User Query
all_questions = [item['question'] for item in dataset]
selected_questions = random.sample(all_questions, min(100, len(all_questions)))

# 4. 构造 ShareGPT 格式
sharegpt_data = []
for i, q in enumerate(selected_questions):
    sharegpt_data.append({
        "id": f"real_rag_test_{i:04d}",
        "conversations": [
            {
                "from": "system",
                "value": f"【系统指令】你是一个专业的 AI 助手。请严格根据以下参考文档回答用户的问题。如果文档中没有答案，请回答“文档中未提及”。\n\n【参考文档】\n{shared_system_prompt}"
            },
            {
                "from": "human",
                "value": q
            },
            {
                "from": "gpt",
                "value": "根据参考文档，" 
            }
        ]
    })

output_file = "sharegpt_real_rag_prefix.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

print(f"🎉 成功生成 {len(sharegpt_data)} 条基于真实数据的 RAG 压测集，已保存至: {output_file}")