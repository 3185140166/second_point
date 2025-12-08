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

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau 
from sklearn.model_selection import train_test_split

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


def generate_lora(input_data, tokenizer, model, projector, device, doc_weights=None, cross_attn_weights=None, use_average=True):
    """
    生成lora：根据参数选择生成平均lora或加权lora
    改进：确保设备一致性，避免跨卡操作
    
    Args:
        input_data: 包含passages_list的输入数据
        tokenizer: 用于编码文本的分词器
        model: 用于编码文本的模型
        projector: 用于生成LoRA的投影器
        device: 设备
        doc_weights: 文档权重 (batch_size, num_passages, 1)
        cross_attn_weights: 交叉注意力权重 (batch_size, num_heads, query_length, key_length)
                           如果提供，将使用交叉注意力权重进行加权
        use_average: 是否生成平均lora，默认为True（生成平均lora）
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
    
    # 对多个lora进行融合
    fused_lora = {}
    num_docs = len(all_loras)
    
    for key in all_loras[0].keys():
        # 堆叠所有文档的LoRA权重
        key_loras = torch.stack([lora[key] for lora in all_loras])  # (num_docs, ...)
        
        if use_average:
            # 生成平均lora：直接求平均
            fused_lora[key] = key_loras.mean(dim=0)
        else:
            # 生成加权lora
            if cross_attn_weights is not None:
                # 使用交叉注意力权重进行加权
                # 稳健的cross_attn_weights处理
                # 1. 对注意力头求平均 (B, num_heads, tgt_len, src_len) -> (B, tgt_len, src_len)
                avg_attn = cross_attn_weights.mean(dim=1)
                # 2. 取第一个查询位置 (B, tgt_len, src_len) -> (B, src_len)
                query_attn = avg_attn[:, 0, :]
                # 3. 确保权重形状正确
                if query_attn.dim() > 1:
                    query_attn = query_attn.squeeze(0)  # (src_len,)
                # 4. 归一化注意力权重
                normalized_weights = F.softmax(query_attn, dim=-1)  # 归一化确保权重和为1
                # 5. 断言src_len == num_docs
                assert query_attn.shape[0] == num_docs, f"src_len ({query_attn.shape[0]}) != num_docs ({num_docs})"
                # 6. 使用交叉注意力权重加权LoRA
                fused_lora[key] = torch.sum(normalized_weights[:, None, None] * key_loras, dim=0)
            elif doc_weights is not None:
                # 使用原始doc_weights进行加权
                weights = doc_weights.squeeze(0).squeeze(-1)  # (1, num_passages, 1) -> (num_passages,)
                fused_lora[key] = torch.sum(weights[:, None, None] * key_loras, dim=0)
            else:
                #  fallback: 如果没有权重，使用平均lora
                fused_lora[key] = key_loras.mean(dim=0)
    
    return fused_lora


def encode_text(text, tokenizer, model, device):
    """
    backbone with gradient attachment for fusion network training
    return (B, hidden_dim) with requires_grad=True
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
        # 获取最后一层的CLS token embedding
        emb = outputs.hidden_states[-1][:,-1,:] #[B,hidden_dim]
    
    # 关键修复：保留值，不保留模型梯度链，但为fusion网络插上梯度
    # 这样模型本体不更新，但fusion网络能有效训练
    emb = emb.detach()  # 移除模型的梯度链
    emb.requires_grad_(True)  # 为fusion网络添加梯度
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

    args.with_cot = True
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

        # 注释：这段代码用于仅将答案部分作为训练标签，非答案部分设为忽略
        # 如果需要更精确的训练，可以取消注释这段代码
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
    fusion_device = get_device(fusion_net)
    

    # 设置随机种子以确保可复现性
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # ------------------------- 超参设置 ------------------------- 
    preheat_epochs = 1
    max_train_epochs = args.dyprag_train_epochs
    eval_every = 1
    patience = 3
    min_delta_f1 = 0.5  # 百分比点
    grad_clip_norm = 1.0
    use_amp = True
    
    # ------------------------- 阶段1：预热训练（Preheat） ------------------------- 
    print(f"\n{'='*80}")
    print(f"STARTING PREHEAT TRAINING (1 epoch) WITH FULL DATA")
    print(f"{'='*80}")
    
    # 预热阶段配置
    preheat_weights = {
        'mse': 10.0,    # MSE权重
        'kl': 0.01      # KL权重（仅最后一个token）
    }
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(preheat_epochs):
        fusion_net.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Preheat Epoch {epoch+1}/{preheat_epochs}")):
            optimizer.zero_grad()
            input_data = batch['input_data'][0]
            model_inputs = batch['model_inputs'][0]
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 1. 编码问题和文档
                q_emb = encode_text(input_data['question'], tokenizer, model, fusion_device).unsqueeze(1)
                passage_embeddings = []
                for passage in input_data['passages_list']:
                    passage_embedding = encode_text(passage, tokenizer, model, fusion_device)
                    passage_embeddings.append(passage_embedding)
                passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)  # (1, num_passages, hidden_dim)
                
                # 2. 融合得到输入嵌入
                input_embeds, doc_weights, cross_attn_weights = fusion_net(q_emb, passage_embeddings)
                fusion_lora = projector(input_embeds.to(device_projector, dtype=torch.float32))
                
                # 3. 生成target lora（平均lora）
                target_lora = generate_lora(input_data, tokenizer, model, projector, fusion_device, doc_weights, cross_attn_weights, use_average=True)
                
                # 4. 计算MSE损失
                mse_loss = torch.tensor(0.0, device=fusion_device, dtype=torch.float32)
                for key in fusion_lora.keys():
                    mse_loss += F.mse_loss(fusion_lora[key].to(fusion_device), target_lora[key].to(fusion_device))
                mse_loss *= preheat_weights['mse']
                
                # 5. 计算KL损失（仅最后一个token）
                # 注入fusion lora
                temp_fusion_outputs = {k: v.to(device) for k, v in fusion_lora.items()}
                delta_inject(model, temp_fusion_outputs)
                lm_outputs = model(**model_inputs)
                delta_remove(model, temp_fusion_outputs)
                del temp_fusion_outputs
                
                # 注入target lora
                temp_target_outputs = {k: v.to(device) for k, v in target_lora.items()}
                delta_inject(model, temp_target_outputs)
                target_lm_outputs = model(**model_inputs)
                delta_remove(model, temp_target_outputs)
                del temp_target_outputs
                
                # 只取最后一个token计算KL
                fusion_logits_last = lm_outputs.logits[:, -1, :]
                target_logits_last = target_lm_outputs.logits[:, -1, :]
                
                kl_loss = F.kl_div(
                    F.log_softmax(fusion_logits_last, dim=-1),
                    F.softmax(target_logits_last, dim=-1),
                    reduction='batchmean'
                ).to(fusion_device) * preheat_weights['kl']
                
                # 总损失
                total_loss = mse_loss + kl_loss
                
                # 保存loss值用于打印
                current_total_loss = total_loss.item()
            
            # 反向传播
            scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_net.parameters(), grad_clip_norm)
            if args.joint_training:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), grad_clip_norm)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
            # 释放内存
            del q_emb, passage_embeddings, input_embeds, doc_weights, cross_attn_weights
            del fusion_lora, target_lora, lm_outputs, target_lm_outputs
            del fusion_logits_last, target_logits_last, mse_loss, kl_loss, total_loss
            
            epoch_loss += current_total_loss
            print(f"  Step {step}, Total Loss: {current_total_loss:.4f}")
        
        print(f"Preheat Epoch {epoch+1} Average Loss: {epoch_loss/len(train_dataloader):.4f}")
    
    # ------------------------- 阶段2：划分训练/验证集 ------------------------- 
    print(f"\n{'='*80}")
    print(f"SPLITTING DATA INTO TRAIN/DEV SETS (80/20)")
    print(f"{'='*80}")
    
    # 划分数据集
    train_samples, dev_samples = train_test_split(
        samples, 
        test_size=0.2, 
        random_state=seed,
        shuffle=True
    )
    
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Dev samples: {len(dev_samples)}")
    
    # 创建训练和验证数据集
    train_dataset = create_lora_passage_dataset(train_samples)
    dev_dataset = create_lora_passage_dataset(dev_samples)
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=TrainingDataCollator(tokenizer, device, model)
    )
    
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=TrainingDataCollator(tokenizer, device, model)
    )
    
    # ------------------------- 阶段3：主训练（Main Training） ------------------------- 
    print(f"\n{'='*80}")
    print(f"STARTING MAIN TRAINING WITH EARLY STOPPING")
    print(f"{'='*80}")
    
    # 早停相关变量
    best_dev_f1 = 0.0
    best_epoch = -1
    patience_counter = 0
    
    # 用于保存checkpoint
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_fusion_checkpoint.pt")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_fusion_checkpoint.pt")
    
    # 定义评估函数
    def evaluate_on_dev():
        """在验证集上评估F1分数"""
        print(f"\n{'='*60}")
        print(f"EVALUATING ON DEV SET")
        print(f"{'='*60}")
        
        fusion_net.eval()
        total_f1 = 0.0
        
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating on Dev")):
            input_data = batch['input_data'][0]
            
            with torch.no_grad():
                # 1. 生成fusion lora
                q_emb = encode_text(input_data['question'], tokenizer, model, fusion_device).unsqueeze(1)
                passage_embeddings = []
                for passage in input_data['passages_list']:
                    passage_embedding = encode_text(passage, tokenizer, model, fusion_device)
                    passage_embeddings.append(passage_embedding)
                passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)
                
                input_embeds, _, _ = fusion_net(q_emb, passage_embeddings)
                fusion_lora = projector(input_embeds.to(device_projector, dtype=torch.float32))
                
                # 2. 注入lora并生成答案
                temp_outputs = {k: v.to(device) for k, v in fusion_lora.items()}
                delta_inject(model, temp_outputs)
                
                # 使用与推理一致的prompt模板生成答案
                generated_answer = predict(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    question=input_data['question'],
                    with_cot=True,
                    passages=input_data['passages_list']
                )
                
                delta_remove(model, temp_outputs)
                del temp_outputs, q_emb, passage_embeddings, input_embeds, fusion_lora
            
            # 3. 计算F1分数
            eval_result = evaluate(generated_answer, input_data['answer'], with_cot=True)
            f1 = float(eval_result['f1'])
            total_f1 += f1
            print(f"  Sample {step}, Generated: {generated_answer[:50]}..., True: {input_data['answer'][:50]}..., F1: {f1:.4f}")
        
        avg_f1 = total_f1 / len(dev_dataloader)
        print(f"Dev Set Average F1: {avg_f1:.4f}")
        return avg_f1
    
    # 主训练循环
    for epoch in range(max_train_epochs):
        print(f"\n{'='*60}")
        print(f"MAIN TRAINING EPOCH {epoch+1}/{max_train_epochs}")
        print(f"{'='*60}")
        
        fusion_net.train()
        if args.joint_training:
            projector.train()
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            input_data = batch['input_data'][0]
            model_inputs = batch['model_inputs'][0]
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 1. 生成fusion lora
                q_emb = encode_text(input_data['question'], tokenizer, model, fusion_device).unsqueeze(1)
                passage_embeddings = []
                for passage in input_data['passages_list']:
                    passage_embedding = encode_text(passage, tokenizer, model, fusion_device)
                    passage_embeddings.append(passage_embedding)
                passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)
                
                input_embeds, _, _ = fusion_net(q_emb, passage_embeddings)
                fusion_lora = projector(input_embeds.to(device_projector, dtype=torch.float32))
                
                # 2. 注入lora并计算LM loss
                temp_outputs = {k: v.to(device) for k, v in fusion_lora.items()}
                delta_inject(model, temp_outputs)
                
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss.to(fusion_device)
                
                # 保存loss值用于打印和累加
                current_lm_loss = lm_loss.item()
                
                delta_remove(model, temp_outputs)
                del temp_outputs
            
            # 反向传播
            scaler.scale(lm_loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_net.parameters(), grad_clip_norm)
            if args.joint_training:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), grad_clip_norm)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
            # 释放内存
            del q_emb, passage_embeddings, input_embeds, fusion_lora, lm_outputs, lm_loss
            
            epoch_loss += current_lm_loss
            print(f"  Step {step}, LM Loss: {current_lm_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average LM Loss: {avg_epoch_loss:.4f}")
        
        # 评估和早停
        if (epoch + 1) % eval_every == 0:
            current_dev_f1 = evaluate_on_dev()
            
            # 保存最佳checkpoint
            if current_dev_f1 > best_dev_f1 + min_delta_f1/100:
                best_dev_f1 = current_dev_f1
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存checkpoint
                if args.joint_training:
                    torch.save({
                        'fusion_net_state_dict': fusion_net.state_dict(),
                        'projector_state_dict': projector.state_dict(),
                        'epoch': epoch + 1,
                        'dev_f1': current_dev_f1,
                    }, best_checkpoint_path)
                else:
                    torch.save({
                        'fusion_net_state_dict': fusion_net.state_dict(),
                        'epoch': epoch + 1,
                        'dev_f1': current_dev_f1,
                    }, best_checkpoint_path)
                print(f"✓ Saved BEST checkpoint at epoch {epoch+1} with Dev F1: {current_dev_f1:.4f}")
            else:
                patience_counter += 1
                print(f"⌛ Patience counter: {patience_counter}/{patience}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                print(f"{'='*60}")
                break
    
    # ------------------------- 保存最后checkpoint ------------------------- 
    if args.joint_training:
        torch.save({
            'fusion_net_state_dict': fusion_net.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'epoch': max_train_epochs,
        }, last_checkpoint_path)
    else:
        torch.save({
            'fusion_net_state_dict': fusion_net.state_dict(),
            'epoch': max_train_epochs,
        }, last_checkpoint_path)
    print(f"✓ Saved LAST checkpoint")
    
    # ------------------------- 训练完成 ------------------------- 
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"Best Epoch: {best_epoch}, Best Dev F1: {best_dev_f1:.4f}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Best Checkpoint: {best_checkpoint_path}")
    print(f"Last Checkpoint: {last_checkpoint_path}")
    print(f"{'='*80}")


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
