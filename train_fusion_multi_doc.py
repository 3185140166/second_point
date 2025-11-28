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
from utils import get_model, evaluate, predict, load_data, read_complete, get_attributes, delta_inject, delta_remove
from encode import get_train_data
from projector import ParameterTranslator
from transformers import DefaultDataCollator
from typing import List, Dict
import torch.nn.functional as F
from safetensors.torch import load_file
# import matplotlib.pyplot as plt
from collections import defaultdict
from fusion import CrossAttentionFusion


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

        

def prepare_training_data_multi_datasets(args, tokenizer, datasets):
    """
    Prepare training data from multiple datasets
    """
    all_training_samples = []
    ignored_id = -100
    max_length = 3000
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        # Update args for current dataset
        args.dataset = dataset_name
        if dataset_name in ("2wikimultihopqa", "hotpotqa"):
            prompt_template.get_fewshot(dataset_name)
            args.with_cot = True  #这里是不是应该也不要呢？
        else:
            args.with_cot = False
        if dataset_name == "popqa":
            num_train_epochs = 2
        else:
            num_train_epochs = 1
        projector = True

        data_list = load_data(dataset_name, None, args.augment_model, projector, data_dir="./data_aug_projector")
        for filename, fulldata in data_list:
            filename = filename.split(".")[0]
            print(f"Collecting data from {filename}")
            data_size = len(fulldata) 
            sample_data = random.sample(range(data_size),int(data_size * args.sample_rate))
            for test_id in tqdm(sample_data):
                data = fulldata[test_id]
                augment = data["augment"] 
                for pid, aug in enumerate(augment):
                    if pid < 3:
                        continue

                    # Get raw prompt_ids using the augmented data
                    raw_prompt_ids = get_train_data(args.augment_model, [aug], tokenizer, args)[2]
                    # 第二个样本的问题
                    question = aug[f"{args.augment_model}_qa"][1]['question']
                    # Process prompt_ids following encode.py's TrainingData class
                    labels = raw_prompt_ids.copy()
                    if len(raw_prompt_ids) > max_length:
                        raw_prompt_ids = raw_prompt_ids[:max_length]
                        labels = labels[:max_length]
                    attention_mask = [1] * len(raw_prompt_ids) + [0] * (max_length - len(raw_prompt_ids))
                    raw_prompt_ids += [pad_token_id] * (max_length - len(raw_prompt_ids))
                    labels += [ignored_id] * (max_length - len(labels))
                 
                    # extract passage_list
                    passages_list = aug['passage'].split(" [sep] ")
                    passages_list = [p.strip() for p in passages_list]
                        
                    all_training_samples.append({
                        'question': question,  
                        "passages_list": passages_list,
                        'input_ids': raw_prompt_ids,
                        'labels': labels,
                        'attention_mask': attention_mask,
                        'file_name': filename,
                        'data_id': test_id,
                        'passage_id': pid,
                        'dataset': dataset_name
                    })
    print(f"\nTotal prepared {len(all_training_samples)} training samples from {len(datasets)} datasets")
    return all_training_samples

def get_device(module):
    return next(module.parameters()).device

class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device, model, fusion_network):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.fusion_network = fusion_network

    
    def __call__(self, examples: List[Dict[str, dict]]) -> Dict[str, torch.Tensor]:
        input_embeds = []
        model_inputs = []
        for example in examples:
            
            with torch.no_grad():

                passages_list = example['passages_list']
                question = example['question']

                fusion_device = get_device(self.fusion_network)
                # ===============================分别编码每个文档==================================
                q_emb = encode_text(question, self.tokenizer, self.model, fusion_device)

                passage_embeddings = []
                for passage in passages_list:
                    passage_embedding = encode_text(passage, self.tokenizer, self.model, fusion_device)
                    passage_embeddings.append(passage_embedding)
                passage_embeddings = torch.cat(passage_embeddings, dim=0).unsqueeze(0)  # (1, num_passages, hidden_dim)
                # ============融合，self-attention============
                final_embedding = self.fusion_network(q_emb.unsqueeze(1), passage_embeddings)  # (B,1,hiddden_dim)  (B,docs,hidden_dim) ->(B,hidden_dim)

                input_embeds.append(final_embedding)

                model_inputs.append({
                    'input_ids': torch.tensor(example['input_ids']).unsqueeze(0).to(self.device),
                    'labels': torch.tensor(example['labels']).unsqueeze(0).to(self.device),
                    'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0).to(self.device)
                })
                    
        return {
            "input_embeds": input_embeds,
            "model_inputs": model_inputs,
        }

def main(args):
    # Define datasets to use
    datasets = args.datasets
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
    training_samples = prepare_training_data_multi_datasets(args, tokenizer, datasets)
    dataset = create_lora_passage_dataset(training_samples)

    fusion_net = CrossAttentionFusion(
        model.config.hidden_size,
        num_heads=8,
        dropout=0.1
    ).to(device_projector)
    fusion_net.train()
    print(f"Fusion network initialized with {sum(p.numel() for p in fusion_net.parameters())} parameters")

    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=TrainingDataCollator(tokenizer, device, model, fusion_network=fusion_net)
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
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False   
    model.eval()

    optimizer = torch.optim.AdamW(fusion_net.parameters(), lr=args.dyprag_learning_rate)

    # Initialize loss tracking
    DYPRAG_TRAIN_EPOCH = args.dyprag_train_epochs
    total_time = 0
    global_step = 0
    for epoch in range(DYPRAG_TRAIN_EPOCH):
        fusion_net.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            start_time = time.time()
            optimizer.zero_grad()
            input_embeds = batch['input_embeds'][0]
            model_inputs = batch['model_inputs'][0]

            with torch.no_grad():
                outputs = projector(input_embeds.to(device_projector).to(torch.float32))
            # Move outputs to model device before injection
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)
            # Language Modeling Loss
            with torch.set_grad_enabled(True):
                # Get language model loss
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss
            # Backward
            lm_loss.backward()
            optimizer.step()
            delta_remove(model, outputs)
            del outputs, lm_outputs
            torch.cuda.empty_cache()
            epoch_loss += lm_loss.item()
            global_step += 1
            print(f"Stage 2: Epoch {epoch}, Step {step}, LM Loss: {lm_loss.item():.4f}")
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch} finished. avg loss = {avg_loss:.4f}")
        save_dir = os.path.join(ROOT_DIR, "fusion", f'{args.model_name}_hidden{args.projector_p}_sample{args.sample_rate}_lr{args.dyprag_learning_rate}_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save funsion to {save_dir}")
        # 同时保存两个网络
        checkpoint = {
            'projector_state_dict': projector.state_dict(),
            'fusion_net_state_dict': fusion_net.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, f"epoch_{epoch}.pt")
        )

    print(f"Finished DyPRAG Training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, )
    parser.add_argument("--max_new_tokens", type=int, )
    parser.add_argument("--datasets", type=list, default=["2wikimultihopqa", "hotpotqa", "popqa", "complexwebquestions"])    # Previous parameterizing settings 
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # DyPRAG training settings
    parser.add_argument("--dyprag_train_epochs", type=int, help="number of DyPRAG training epochs")
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--dyprag_learning_rate", type=float, default=1e-5)
    parser.add_argument("--projector_p", type=int, default=32)
    parser.add_argument("--inference_epoch", type=int, required=True)
    parser.add_argument("--projector_path", type=str, required=True)
    parser.add_argument("--augment_model", type=str, default=None)
    args = parser.parse_args()
    
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
