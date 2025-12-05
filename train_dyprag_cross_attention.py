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
        answer_start_idx = -1
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
            print("eorror: not answer token")
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

def eval_on_dev(fusion_net, projector, model, tokenizer, dev_dataloader, device, fusion_device, generation_config, args):
    """dev"""
    fusion_net.eval()
    dev_loss = 0.0
    results = []
   
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Evaluating"):
            input_data = batch['input_data'][0]
            model_inputs = batch['model_inputs'][0]

            q_emb = encode_text(input_data['question'], tokenizer, model, fusion_device).unsqueeze(1)
            passage_embeddings = []
            for passage in input_data['passages_list']:
                passage_embedding = encode_text(passage, tokenizer, model, fusion_device)
                passage_embeddings.append(passage_embedding)
            passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)
            input_embeds, doc_weights, cross_attn_weights = fusion_net(q_emb, passage_embeddings)
            outputs = projector(input_embeds.to(fusion_device, dtype=torch.float32))
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)
            
            lm_outputs = model(**model_inputs)
            lm_loss = lm_outputs.loss.to(fusion_device)
            
            # 计算完整损失（与训练时一致）
            weight_reg_loss = -torch.mean(torch.var(doc_weights, dim=1)) * 0.1
            diversity_loss = torch.mean(torch.sum(cross_attn_weights * torch.log(cross_attn_weights + 1e-10), dim=2)) * 0.05
            total_loss = lm_loss + weight_reg_loss + diversity_loss
            
            dev_loss += total_loss.item()
            
            #==================add===================
            dataset = input_data['dataset']
            if dataset in loaded_fewshots:
                prompt_template.fewshot = loaded_fewshots[dataset]
            text =  predict(model, tokenizer, generation_config, 
                             input_data['question'], args.with_cot, 
                             passages=input_data['passages_list'])
            metrics = evaluate(text, input_data['answer'], args.with_cot)
            results.append(metrics)
            
            delta_remove(model, outputs)
            del outputs, lm_outputs
            del input_embeds, doc_weights, cross_attn_weights
            del lm_loss, weight_reg_loss, diversity_loss, total_loss
            del q_emb, passage_embeddings
            torch.cuda.empty_cache()
        avg_dev_loss = dev_loss / len(dev_dataloader)
        #===================计算平均指标===================
    avg_metrics = {}
    for met in ["em", "f1", "prec", "recall"]:
        values = [float(r[met]) for r in results if met in r]
        avg_metrics[met] = round(sum(values) / len(values) if values else 0, 4)
    return avg_dev_loss, avg_metrics

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
    train_samples, dev_samples = train_test_split(
        samples, test_size= 0.2, random_state=42
    )
    train_dataset = create_lora_passage_dataset(train_samples)
    dev_dataset = create_lora_passage_dataset(dev_samples)

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
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False,
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
    projector.load_state_dict(torch.load(projector_path, map_location=model.device)['model_state_dict'])
    
    if args.joint_training:
        # 联合训练：projector设为train模式，开启梯度更新
        projector.train()
        for p in projector.parameters():
            p.requires_grad = True
        print("Joint training enabled: projector will be trained with learning rate", args.projector_learning_rate)
    else:
        # 固定projector：设为eval模式，关闭梯度更新
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
        print("Fixed projector: only fusion network will be trained")
    
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
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=DYPRAG_TRAIN_EPOCH * len(train_dataloader),
        eta_min=min(args.dyprag_learning_rate * 0.01, args.projector_learning_rate * 0.01)
    )
    # history_train
    history = {
        'train_loss': [],
        'dev_loss': [],
        'dev_em': [],
        'dev_f1': [],
        'dev_prec': [],
        'dev_recall': []
    }
    best_f1 = -float('inf')
    best_epoch = -1

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
            # Move outputs to model device before injection
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)
            # Language Modeling Loss
            with torch.set_grad_enabled(True):
                # Get language model loss
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss.to(fusion_device)
            
            # 文档权重正则化损失：鼓励模型学习不同文档的重要性，避免平均化
            weight_reg_loss = -torch.mean(torch.var(doc_weights, dim=1)) * 0.1
            
            # 多样性正则化损失：鼓励模型关注多个文档，避免过度依赖单一文档
            diversity_loss = torch.mean(torch.sum(cross_attn_weights * torch.log(cross_attn_weights + 1e-10), dim=2)) * 0.05
            
            # 总损失
            total_loss = lm_loss + weight_reg_loss + diversity_loss
            
            # Backward
            total_loss.backward()

            if global_step <= 5:
                for p in fusion_net.parameters():
                    if p.grad is not None:
                        print("fusion grad norm:", p.grad.norm().item())
                        break
                    else:
                        print("梯度是None")
            optimizer.step()
            scheduler.step()
            delta_remove(model, outputs)
            
            # 先保存所有需要的值，再释放内存
            current_total_loss = total_loss.item()
            current_lm_loss = lm_loss.item()
            current_weight_reg_loss = weight_reg_loss.item()
            current_diversity_loss = diversity_loss.item()
            
            # 释放GPU内存：删除不再需要的变量
            del outputs, lm_outputs
            del input_embeds, doc_weights, cross_attn_weights
            del lm_loss, weight_reg_loss, diversity_loss, total_loss
            del q_emb, passage_embeddings
            
            torch.cuda.empty_cache()
            epoch_loss += current_total_loss
            global_step += 1
            if step%10 ==0:
                print(f"  Step {step}, Total Loss: {current_total_loss:.4f}, LM Loss: {current_lm_loss:.4f}, Weight Reg Loss: {current_weight_reg_loss:.4f}, Diversity Loss: {current_diversity_loss:.4f}")
        avg_train_loss = epoch_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch} finished. avg train loss = {avg_train_loss:.4f}")
        ##============dev===================
        print("\nValidating...")
        avg_dev_loss, avg_metrics = eval_on_dev(
            fusion_net, projector, model, tokenizer,dev_dataloader, 
            device, fusion_device, generation_config, args
        )
        history['dev_loss'].append(avg_dev_loss)
        history['dev_em'].append(avg_metrics['em'])
        history['dev_f1'].append(avg_metrics['f1'])
        history['dev_prec'].append(avg_metrics['prec'])
        history['dev_recall'].append(avg_metrics['recall'])
        
        print(f"\nEpoch {epoch} Dev Results:")
        print(f"  Loss:   {avg_dev_loss:.4f}")
        print(f"  EM:     {avg_metrics['em']:.4f}")
        print(f"  F1:     {avg_metrics['f1']:.4f}")
        print(f"  Prec:   {avg_metrics['prec']:.4f}")
        print(f"  Recall: {avg_metrics['recall']:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Current LR: {current_lr:.6f}")

        fusion_net.train()  # 记得切换回 train 模式

        #=================save per 5 epoch============
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{epoch}_f1_{avg_metrics['f1']}.pt")
            torch.save({
                'fusion_net_state_dict': fusion_net.state_dict(),
                'projector_state_dict': projector.state_dict()
            }, ckpt_path)
            print(f"✓ Saved checkpoint: {ckpt_path}")
        #=================save_best===================
        if avg_metrics['f1'] > best_f1:
            best_f1 = avg_metrics['f1']
            best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'fusion_net_state_dict': fusion_net.state_dict(),
                'projector_state_dict': projector.state_dict()
            }, best_path)
            print(f"🌟 Saved best model (F1={best_f1:.4f}): {best_path}")
        #==================save_history================
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'history': history,
                'best_epoch': best_epoch,
                'best_f1': best_f1,
                'args': vars(args)
            }, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Best Epoch: {best_epoch} (F1={best_f1:.4f})")
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
    parser.add_argument("--joint_training", type=bool, default=True, help="Whether to train projector jointly")
    args = parser.parse_args()
    
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
