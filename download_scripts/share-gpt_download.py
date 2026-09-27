# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from datasets import load_dataset
# import json

# from huggingface_hub import hf_hub_download

# repo_id = "anon8231489123/ShareGPT_Vicuna_unfiltered"
# filename = "ShareGPT_V3_unfiltered_cleaned_split.json" 
# output_file = "sharegpt_1000_final.json"
# try:
#     local_path = hf_hub_download(
#         repo_id=repo_id, 
#         filename=filename, 
#         repo_type="dataset"
#     )
# except Exception as e:
#     print(f"fail")
#     exit(1)
import json

# 配置输入和输出文件名
input_file = "ShareGPT_V3_unfiltered_cleaned_split.json"   # 替换为你的原始 JSON 文件名
output_file = "files/sharegpt-1000.json"   # 替换为你想保存的新文件名

print(f"正在读取 {input_file} ...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据共 {len(data)} 条，正在截取前 1000 条...")
# 截取前 1000 条
data_top_1000 = data[:1000]

print(f"正在保存到 {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    # ensure_ascii=False 保证中文正常显示，indent=2 让格式美观
    json.dump(data_top_1000, f, ensure_ascii=False, indent=2)

print("✅ 处理完成！")