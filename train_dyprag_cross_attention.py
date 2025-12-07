# Stage 2 (Phase 2): Train Fusion with Frozen Projector
import os
import gc
import json
import argparse
import torch
import random
from tqdm import tqdm
import time
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, get_attributes, delta_inject, delta_remove, get_gloden_data
from projector import ParameterTranslator
from transformers import DefaultDataCollator
from typing import List, Dict
import torch.nn.functional as F
from safetensors.torch import load_file
# import matplotlib.pyplot as plt
from collections import defaultdict
from fusion import CrossAttentionFusion
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau 

unique_datasets = ["2wikimultihopqa", "hotpotqa"]
loaded_fewshots = {}
for ds in unique_datasets:
    prompt_template.get_fewshot(ds)
    loaded_fewshots[ds] = prompt_template.fewshot

def create_lora_passage_dataset(lora_passage_pairs):
    """
    Create a PyTorch dataset from LoRA-passage pairs
    """
    from torch.utils.data import Dataset
    
    class LoRAPassageDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
            
        def __len__(self):
            return len(self.pairs)
            
        def __getitem__(self, idx):
            return self.pairs[idx]
    
    return LoRAPassageDataset(lora_passage_pairs)

def load_model_checkpoint(fusion_net, projector, checkpoint_path, device):
    """
    Load model checkpoint for fusion_net and projector
    
    Args:
        fusion_net: CrossAttentionFusion model to load weights into
        projector: ParameterTranslator model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load weights onto
        
    Returns:
        fusion_net: Loaded CrossAttentionFusion model
        projector: Loaded ParameterTranslator model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load fusion_net weights
    if 'fusion_net_state_dict' in checkpoint:
        fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])
        print(f"✓ Loaded fusion_net weights from {checkpoint_path}")
    else:
        print(f"⚠ No fusion_net_state_dict found in {checkpoint_path}")
    
    # Load projector weights
    if 'projector_state_dict' in checkpoint:
        projector.load_state_dict(checkpoint['projector_state_dict'])
        print(f"✓ Loaded projector weights from {checkpoint_path}")
    else:
        print(f"⚠ No projector_state_dict found in {checkpoint_path}")
    
    return fusion_net, projector


def generate_weighted_lora(input_data, tokenizer, model, projector, device, doc_weights, cross_attn_weights=None):
    """
    生成加权lora：对每个文档生成lora，然后根据权重进行加权平均
    
    Args:
        input_data: 包含passages_list的输入数据
        tokenizer: 用于编码文本的分词器
        model: 用于编码文本的模型
        projector: 用于生成LoRA的投影器
        device: 设备
        doc_weights: 文档权重 (batch_size, num_passages, 1)
        cross_attn_weights: 交叉注意力权重 (batch_size, num_heads, query_length, key_length)
                           如果提供，将使用交叉注意力权重进行加权
    """
    all_loras = []
    
    # 对每个文档生成单独的lora
    for passage in input_data['passages_list']:
        # 编码单个文档，得到文档embedding
        single_passage_emb = encode_text(passage, tokenizer, model, device)
        
        # 通过projector生成单个lora
        with torch.no_grad():
            single_lora = projector(single_passage_emb.to(device, dtype=torch.float32))
        
        all_loras.append(single_lora)
    
    # 对多个lora进行加权融合
    weighted_lora = {}
    num_docs = len(all_loras)
    
    for key in all_loras[0].keys():
        # 堆叠所有文档的LoRA权重
        key_loras = torch.stack([lora[key] for lora in all_loras])  # (num_docs, ...)
        
        if cross_attn_weights is not None:
            # 使用交叉注意力权重进行加权
            # 1. 对注意力头求平均 (batch_size, 1, query_length, key_length)
            avg_attn_weights = cross_attn_weights.mean(dim=1)  # 对多头注意力求平均
            # 2. 只使用第一个查询位置的注意力权重 (batch_size, key_length)
            query_attn_weights = avg_attn_weights.squeeze(1).squeeze(1)  # (batch_size, num_passages)
            # 3. 确保权重形状正确
            query_attn_weights = query_attn_weights.squeeze(0)  # (num_passages,)
            # 4. 归一化注意力权重
            normalized_weights = F.softmax(query_attn_weights, dim=0)  # 归一化确保权重和为1
            # 5. 使用交叉注意力权重加权LoRA
            weighted_lora[key] = torch.sum(normalized_weights[:, None, None] * key_loras, dim=0)
        else:
            # 使用原始doc_weights进行加权
            weights = doc_weights.squeeze(0).squeeze(-1)  # (1, num_passages, 1) -> (num_passages,)
            weighted_lora[key] = torch.sum(weights[:, None, None] * key_loras, dim=0)
    
    return weighted_lora


def encode_text(text, tokenizer, model, device):
    """
    backbone
    return (B, hidden_dim)
    
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3000
    ).to(device)

    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            inputs.attention_mask,
            output_hidden_states=True
            )
        emb = outputs.hidden_states[-1][:,-1,:] #[B,hidden_dim]
    return emb


def prepare_training_data_multi_datasets(args, tokenizer, dataset_path):
    """
    Prepare training data from multiple datasets
    """
    all_training_samples = []
    ignored_id = -100
    max_length = 3000
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    json_file = os.path.join(ROOT_DIR, dataset_path)
    print(f"\nLoading data from: {json_file}")
    with open(json_file, 'r') as fin:
        data_list = json.load(fin)
    print(f"Loaded {len(data_list)} samples from dataset")

    args.with_cot = False
    data_size = len(data_list)
    sample_data = random.sample(range(data_size),int(data_size * args.sample_rate))
    cnt = 0
    for test_id in tqdm(sample_data):
        data = data_list[test_id]
        dataset = data['dataset']
        if dataset in loaded_fewshots:
            prompt_template.fewshot = loaded_fewshots[dataset]
        # Get raw prompt_ids using the augmented data
        raw_prompt_ids = get_gloden_data(data, tokenizer, args)[0]
        question = data['question']

        chat_template = getattr(tokenizer, 'chat_template', "")
        start_tokens = tokenizer("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)['input_ids']
        if "<|start_header_id|>assistant<|end_header_id|>" not in chat_template:
            start_tokens = tokenizer("[/INST]", add_special_tokens=False)["input_ids"]

        # Process prompt_ids following encode.py's TrainingData class
        labels = raw_prompt_ids.copy()
        # answer_start_idx = -1
        if len(raw_prompt_ids) > max_length:
            raw_prompt_ids = raw_prompt_ids[:max_length]
            labels = labels[:max_length]
        attention_mask = [1] * len(raw_prompt_ids) + [0] * (max_length - len(raw_prompt_ids))
        raw_prompt_ids += [pad_token_id] * (max_length - len(raw_prompt_ids))
        labels += [ignored_id] * (max_length - len(labels))

        for i in range(len(raw_prompt_ids) - len(start_tokens), -1, -1):
            if raw_prompt_ids[i:i+len(start_tokens)] == start_tokens:
                answer_start_idx = i + len(start_tokens)
                break
        if answer_start_idx == -1:
            print("error: not answer token")
        else:
            for i in range(len(labels)):
                if i < answer_start_idx or raw_prompt_ids[i] == pad_token_id:
                    labels[i] = ignored_id
        # extract passage_list
        passages_list = data['golden_passages']
        passages_list = [p.strip() for p in passages_list]        
        all_training_samples.append({
            'question': question,  
            "passages_list": passages_list,
            'answer': data['answer'],
            'input_ids': raw_prompt_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'data_id': test_id,
            'dataset': dataset
            })
        cnt += 1
        if cnt< 3:
            print(all_training_samples)
    print(f"\nTotal prepared {len(all_training_samples)} samples")
    return all_training_samples

def get_device(module):
    return next(module.parameters()).device

class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.model = model

    
    def __call__(self, examples: List[Dict[str, dict]]) -> Dict[str, torch.Tensor]:
        input_data = []
        model_inputs = []
        for example in examples:

            input_data.append({
                "question": example['question'],
                "passages_list": example['passages_list'],
                'answer': example['answer'],
                'dataset': example['dataset']
            })

            model_inputs.append({
                    'input_ids': torch.tensor(example['input_ids']).unsqueeze(0).to(self.device),
                    'labels': torch.tensor(example['labels']).unsqueeze(0).to(self.device),
                    'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0).to(self.device)
                })
                    
        return {
            "input_data": input_data,
            "model_inputs": model_inputs,
        }



def main(args):
    # Define datasets to use
    DYPRAG_TRAIN_EPOCH = args.dyprag_train_epochs
    dataset_path = args.dataset_path
    # We use two GPUs for training (one A100 is not enough for LLaMA-8B)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #model, tokenizer, embedding
    device_projector = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # fusion and projector
    print(f"Model device: {device}")
    print(f"Projector device: {device_projector}")
    
    base_model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create base LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    # Convert to PeftModel
    model = get_peft_model(base_model, peft_config)
    # Prepare training data from all datasets
    samples = prepare_training_data_multi_datasets(args, tokenizer, dataset_path)
    # 不划分数据集，直接使用完整的训练数据，与dyprag的训练策略保持一致
    train_dataset = create_lora_passage_dataset(samples)


    fusion_net = CrossAttentionFusion(
        model.config.hidden_size,
        num_heads=8,
        dropout=0.1
    ).to(device_projector)
    print(f"Fusion network initialized with {sum(p.numel() for p in fusion_net.parameters())} parameters")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=TrainingDataCollator(tokenizer, device, model)
    )

    print(f"initialize projector with {args.projector_p} hidden layers")
    # Initialize projector
    projector_path = os.path.join(ROOT_DIR, "projector", args.projector_path, f"epoch_{args.inference_epoch-1}.pt")
    projector = ParameterTranslator(
        ["down_proj", "up_proj", "gate_proj"],
        list(range(model.config.num_hidden_layers)),
        model.config.hidden_size,
        model.config.intermediate_size,
        args.lora_rank,
        args.projector_p,
    ).to(device_projector)
    projector.load_state_dict(torch.load(projector_path, map_location=device_projector)['model_state_dict'])
    
    # 根据joint_training参数决定是否冻结projector
    if args.joint_training:
        # 联合训练：projector也参与训练
        projector.train()
        for p in projector.parameters():
            p.requires_grad = True
        print("✓ Joint training mode: Both fusion_net and projector will be trained")
    else:
        # 固定projector：只训练fusion_net
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
        print("✓ Projector parameters frozen and set to eval mode")
        print("✓ Only fusion_net will be trained, aligning with average lora")
    
    model.eval()

    # 根据是否联合训练创建不同的优化器
    if args.joint_training:
        # 联合训练：为fusion_net和projector设置不同的学习率
        optimizer = torch.optim.AdamW([
            {'params': fusion_net.parameters(), 'lr': args.dyprag_learning_rate},
            {'params': projector.parameters(), 'lr': args.projector_learning_rate}
        ])
    else:
        # 固定projector：只训练fusion_net
        optimizer = torch.optim.AdamW(fusion_net.parameters(), lr=args.dyprag_learning_rate)
    

    checkpoint_dir = os.path.join(
        ROOT_DIR, "fusion", 
        f'{args.model_name}_hidden{args.projector_p}_sample{args.sample_rate}_lr{args.dyprag_learning_rate}'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Initialize loss tracking
    total_time = 0
    global_step = 0
    fusion_device = get_device(fusion_net)
    for epoch in range(DYPRAG_TRAIN_EPOCH):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.dyprag_train_epochs - 1}")
        print(f"{'='*60}")
        fusion_net.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            start_time = time.time()
            optimizer.zero_grad()
            input_data = batch['input_data'][0]
            model_inputs = batch['model_inputs'][0]

            #================================encode========================== 
            q_emb = encode_text(input_data['question'], tokenizer, model, fusion_device).unsqueeze(1)
            passage_embeddings = []
            for passage in input_data['passages_list']:
                passage_embedding = encode_text(passage, tokenizer, model, fusion_device)
                passage_embeddings.append(passage_embedding)
            passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)  # (1, num_passages, hidden_dim) 
            #================================fusion===========================
            input_embeds, doc_weights, cross_attn_weights = fusion_net(q_emb, passage_embeddings)   # (B,hidden_dim), (B, num_passages, 1), (B, 1, num_passages)
            outputs = projector(input_embeds.to(device_projector, dtype=torch.float32))
            
            # =================================生成加权平均lora===========================
            # 使用fusion网络生成的动态文档权重和交叉注意力权重生成加权平均lora
            weighted_lora = generate_weighted_lora(input_data, tokenizer, model, projector, fusion_device, doc_weights, cross_attn_weights)
            
            # =================================计算对齐损失===========================
            # 对齐损失：fusion_lora与加权平均lora的MSE损失（软约束）
            mse_loss = torch.tensor(0.0, device=fusion_device, dtype=torch.float32)  
            for key in outputs.keys():
                mse_loss += F.mse_loss(outputs[key].to(fusion_device), weighted_lora[key].to(fusion_device))
            mse_loss *= 100
            
            # Move outputs to model device before injection
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)
            # Language Modeling Loss
            with torch.set_grad_enabled(True):
                # Get language model loss
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss.to(fusion_device)
            delta_remove(model, outputs)
            del outputs
            # =================================计算输出分布对齐损失（KL散度）===========================
            # 使用加权lora生成输出分布，作为对齐目标
            # 对于内容相似的文档，输出分布应该相似
            with torch.no_grad():
                # 注入加权lora
                weighted_outputs = {k: v.to(device) for k, v in weighted_lora.items()}
                delta_inject(model, weighted_outputs)
                # 生成输出
                weighted_lm_outputs = model(**model_inputs)
                weighted_logits = weighted_lm_outputs.logits
                # 移除lora
                delta_remove(model, weighted_outputs)
            
            # 计算KL散度损失（输出分布对齐）
            # 只保留输出分布对齐，不约束中间参数
            kl_loss = F.kl_div(
                F.log_softmax(lm_outputs.logits, dim=-1),
                F.softmax(weighted_logits, dim=-1),
                reduction='batchmean'
            ).to(fusion_device) * 0.01  # 输出分布对齐权重
            
            # =================================计算权重多样性损失===========================
            # 增加权重多样性损失：鼓励文档权重分布更分散，避免平均
            #weight_entropy = -torch.sum(doc_weights * torch.log(doc_weights + 1e-10), dim=1).mean()
            # 提高多样性损失权重，强调权重分散性
            #diversity_loss = 0.1 * weight_entropy 
            
            # 总损失：以语言模型损失为主，辅以输出分布对齐和权重多样性约束
            # 去掉了MSE损失（中间参数对齐），只保留输出分布对齐
            total_loss = lm_loss + kl_loss +  mse_loss #+ diversity_loss
            
            # Backward
            total_loss.backward()

            # 只在训练开始时打印少量梯度信息
            if global_step <= 5:
                for p in fusion_net.parameters():
                    if p.grad is not None:
                        print(f"[Debug] Global Step {global_step}: fusion grad norm: {p.grad.norm().item():.6f}")
                        break
                    else:
                        print(f"[Debug] Global Step {global_step}: No gradients found")
            optimizer.step()
            
            # 更新全局步数
            global_step += 1
            
            # 先保存所有需要的值，再释放内存
            current_total_loss = total_loss.item()
            current_lm_loss = lm_loss.item()
            current_kl_loss = kl_loss.item()
            # current_diversity_loss = diversity_loss.item()
            # current_weight_entropy = weight_entropy.item()
            current_mse_loss =mse_loss.item()
            
            # 释放GPU内存：删除不再需要的变量
            del lm_outputs, weighted_lora, weighted_lm_outputs, weighted_logits
            del input_embeds, doc_weights, cross_attn_weights
            del lm_loss, kl_loss, total_loss, mse_loss #, diversity_loss
            del q_emb, passage_embeddings
            
            torch.cuda.empty_cache()
            print(f"  Step {step}, Total Loss: {current_total_loss:.4f}, LM Loss: {current_lm_loss:.4f}, KL Loss: {current_kl_loss:.6f}, Mse Loss:{current_mse_loss:.6f}")

    if args.joint_training:
    # 联合训练：保存fusion_net和projector的权重
        torch.save({
            'fusion_net_state_dict': fusion_net.state_dict(),
            'projector_state_dict': projector.state_dict(),
        }, os.path.join(checkpoint_dir, f"joint_fusion_projector.pt"))
        print(f"✓ Saved joint fusion-projector checkpoint: joint_fusion_projector.pt")
    else:
    # 非联合训练：只保存fusion_net的权重（projector固定）
        torch.save({
            'fusion_net_state_dict': fusion_net.state_dict(),
        }, os.path.join(checkpoint_dir, f"fusion_1207.pt"))
        print(f"✓ Saved fusion-only checkpoint: fusion.pt")
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, )
    parser.add_argument("--max_new_tokens", type=int, )
    parser.add_argument("--dataset_path", type=str, default='dataset/fusion_dataset.json') #, "popqa", "complexwebquestions"])    # Previous parameterizing settings 
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # DyPRAG training settings
    parser.add_argument("--dyprag_train_epochs", type=int, help="number of DyPRAG training epochs")
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--dyprag_learning_rate", type=float, default=1e-5)
    parser.add_argument("--projector_learning_rate", type=float, default=1e-6, help="Learning rate for projector in joint training")
    parser.add_argument("--projector_p", type=int, default=32)
    parser.add_argument("--inference_epoch", type=int, required=True)
    parser.add_argument("--projector_path", type=str, required=True)
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--joint_training", action="store_true", default=False, help="Whether to train projector jointly")
    args = parser.parse_args()
    
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
