import os
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from root_dir_path import ROOT_DIR
from utils import load_data

def normalize_answer(answer):
    if isinstance(answer, list):
        return answer[0] if len(answer) > 0 else ""
    else:
        return answer
    
def extract_fusion_data(args, datasets, output_file):

    all_data = []
    stats = defaultdict(int)
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*70}")
        
        # 加载数据
        data_list = load_data(
            dataset_name, None, args.augment_model, projector=True, 
            data_dir="./data_aug_projector"
        )
        
        for filename, fulldata in data_list:
            filename = filename.split(".")[0]
            print(f"Processing file: {filename} ({len(fulldata)} samples)")
            
            data_size = len(fulldata)
            
            for test_id in tqdm(range(data_size), desc=f"Extracting from {filename}"):
                data = fulldata[test_id]
                
                # ============ 提取字段 ============
                question = data['question']
                answer = data['answer']
                label = normalize_answer(answer)
                golden_passages = data['golden_passages']                
                # ============ 构造样本 ============
                sample = {
                    'dataset': dataset_name,
                    'filename': filename,
                    'data_id': test_id,
                    'question': question,
                    'answer': answer,  # ← 保证是字符串
                    'golden_passages': golden_passages,
                    'label': label
                }
                
                all_data.append(sample)
                
                # 统计
                stats[dataset_name] += 1

    
    # ============ 保存到 JSON ============
    print(f"\n{'='*70}")
    print(f"Saving to JSON: {output_file}")
    print(f"{'='*70}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Successfully saved {len(all_data)} samples to:")
    print(f"   {output_file}")
    
    # ============ 统计信息 ============
    print(f"\n{'='*70}")
    print("Statistics:")
    print(f"{'='*70}")
    for dataset_name in datasets:
        count = stats[dataset_name]
        if count > 0:
            print(f"  {dataset_name:20s}: {count:5d} samples")
    print(f"  {'Total':20s}: {len(all_data):5d} samples")
    print(f"{'='*70}\n")
    
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data for fusion network training")
    parser.add_argument("--datasets", type=list, default=["2wikimultihopqa", "hotpotqa"],
                       help="Datasets to process")
    parser.add_argument("--output_file", type=str, 
                       default=os.path.join(ROOT_DIR, "data", "fusion_training_data.json"),
                       help="Output JSON file path")
    parser.add_argument("--augment_model", type=str, default=None)
    args = parser.parse_args()
    print(args)
    
    # 运行提取
    extract_fusion_data(args, args.datasets, args.output_file)
