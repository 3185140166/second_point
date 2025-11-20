import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel
from projector import ParameterTranslator
import prompt_template
import random
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, load_ragtruth, read_complete, delta_inject, delta_remove
from peft import LoraConfig, TaskType, get_peft_model

RAGTRUTH_PROMPT_W_PASSAGES = """You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"""

def predict(model, tokenizer, generation_config, question, passages = None):
    model.eval()
    prompt = RAGTRUTH_PROMPT_W_PASSAGES.format(question=question, passages=passages)
    messages = [{
        "role": "user",
        "content": prompt,
    }]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    input_len = len(inputs)
    input_ids = torch.tensor(inputs).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids.unsqueeze(0), 
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text, prompt

def main(args):
    data_list = load_ragtruth(args.dataset, args.data_type)
    
    # Random sample 100 data
    random.seed(42)
    idxs = random.sample(range(len(data_list)), 100)
    data_list = [data_list[i] for i in idxs]
   
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    
    # No CoT
    args.with_cot = False
    cot_name = "cot" if args.with_cot else "direct"
    
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    os.makedirs(output_root_dir, exist_ok=True)
    with open(os.path.join(output_root_dir, "config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    predict_file = os.path.join(output_root_dir, args.predict_file)
    ret = []
    if args.inference_method == "dyprag_combine" or args.inference_method == "dyprag":
        peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        projector_path = os.path.join(ROOT_DIR, "projector", args.projector_path, f"epoch_{args.inference_epoch-1}.pt")
        projector = ParameterTranslator(
            ["down_proj", "up_proj", "gate_proj"],
            list(range(model.config.num_hidden_layers)),
            model.config.hidden_size,
            model.config.intermediate_size,
            args.lora_rank,
            args.projector_p
        ).to(model.device)
        projector.load_state_dict(torch.load(projector_path, map_location=model.device)['model_state_dict'])
        projector.eval()
    for test_id, data in tqdm(enumerate(data_list), total=len(data_list)):
        assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"
        question = data["source_info"]["question"]
        passages = data["source_info"]["passages"]
        def get_pred(model, psgs):
            text, prompt = predict(model, tokenizer, generation_config, 
                                    question, 
                                    passages=psgs)
            pred = {
                "test_id": test_id, 
                "question": question, 
                "passages": passages,
                "prompt": prompt,
                "text": text,
            }
            return pred
        if args.inference_method == "no_icl":
            ret.append(get_pred(model, psgs=None))
        elif args.inference_method == "icl":
            ret.append(get_pred(model, psgs=passages))
        elif args.inference_method == "dyprag" or args.inference_method == "dyprag_combine":
            passages_list = [passages.split("passage")[i+1][3:] for i in range(3)]
            all_deltas = []
            tokenizer.pad_token = tokenizer.eos_token
            for passage in passages_list:
                while passage[-1] == "\n" or passage[-1] == ".":
                    passage = passage[:-1]
                tokens = tokenizer.encode(
                    passage,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    output = model(tokens, output_hidden_states=True)
                    input_embeds = output.hidden_states[-1][:,-1,:]
                    outputs = projector(input_embeds)
                    all_deltas.append(outputs)
            merged_deltas = {}
            for key in all_deltas[0].keys():
                merged_deltas[key] = torch.stack([delta[key] for delta in all_deltas]).mean(dim=0)
            delta_inject(model, merged_deltas)
            ret.append(get_pred(model, psgs=None if args.inference_method == "dyprag" else passages))
            delta_remove(model, merged_deltas)            
            del all_deltas, merged_deltas
            torch.cuda.empty_cache()
            gc.collect()
        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "dyprag", "dyprag_combine", "no_icl"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    # DyPRAG
    parser.add_argument("--projector_p", type=int, default=32)
    parser.add_argument("--inference_epoch", type=int, default=0)
    parser.add_argument("--projector_path", type=str, default=None)
    parser.add_argument("--predict_file", type=str, default="predict.json")
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)