# Adopted from PRAG: https://github.com/oneal2000/PRAG
import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

from utils import get_model, model_generate
from root_dir_path import ROOT_DIR

random.seed(42)


def load_popqa(data_path, dataset, mode):
    data_path = os.path.join(data_path, dataset +'/'+ mode + "/total.json")
    with open(data_path, "r") as fin:
        data = json.load(fin)
    return {"total": data}


def load_complexwebquestions(data_path, dataset, mode):
    data_path = os.path.join(data_path, dataset +'/'+ mode + "/total.json")
    with open(data_path, "r") as fin:
        data = json.load(fin)
    return {"total": data}


def load_2wikimultihopqa(data_path, dataset, mode):
    file_name = ['bridge_comparison','comparison','compositional','inference','total']
    # file_name = ['comparison']
    ret = {}
    for file in file_name:  
        with open(os.path.join(data_path, dataset +'/'+ mode +"/"+ file+".json"), "r") as fin:
            data = json.load(fin)
            ret[file] = data
    return ret


def load_hotpotqa(data_path, dataset, mode):
    file_name = ['bridge','comparison','total']
    ret = {}
    for file in file_name:  
        with open(os.path.join(data_path, dataset +'/'+ mode +"/"+ file+".json"), "r") as fin:
            data = json.load(fin)
            ret[file] = data
    return ret



def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)


qa_prompt_template = "I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"

# 跨文档QA生成模板，要求问题需要整合多个文档的信息
trans_document_qa_prompt_template = (
    "I will provide multiple passages of text. Your task is to generate five different questions that can only be answered by synthesizing information from multiple passages. "
    "Each question must require combining information across all passages. Do not create questions that can be answered using only a single passage.\n\n"
    "To help you think more broadly, here are some types of questions you can consider:\n"
    "- **Comparison questions** (e.g., comparing entities or events mentioned in different passages)\n"
    "- **Cause-and-effect questions** (e.g., identifying how one event described in one passage influences or results in another event in a different passage)\n"
    "- **Temporal reasoning questions** (e.g., constructing a timeline or understanding chronological dependencies)\n"
    "- **Entity synthesis questions** (e.g., asking about a person/organization/concept that appears in multiple passages with different attributes)\n"
    "- **Theme or topic integration** (e.g., identifying common themes, contrasting perspectives, or overarching insights across documents)\n\n"
    "Each question should be answerable using the combined information provided in the set of passages. For each question, provide both a short answer and a full sentence answer.\n\n"
    "You must follow this format strictly:\n"
    "[\n"
    "    {{\n"
    "        \"question\": \"What is the capital of France?\",\n"
    "        \"answer\": \"Paris\",\n"
    "        \"full_answer\": \"The capital of France is Paris.\"\n"
    "    }},\n"
    "    ... (at least five in total)\n"
    "]\n\n"
    "Only output the list in this exact format.\n\n"
    "Passages:\n"
    "{passage}"
)

def fix_qa(qa, num_qa=3): 
     # 3个问题 
    if isinstance(qa, list): 
         # 定义示例问题的模式或关键词
        example_questions = [
             "What is the capital of France?",
        ]
         
         # 过滤掉示例问题
        filtered_qa = []
        for item in qa:
            # 检查是否是有效的字典并且包含必要的键
            if isinstance(item, dict) and "question" in item:
                # 检查是否是示例问题
                is_example = False
                for example in example_questions:
                    if example.lower() in item["question"].lower():
                     is_example = True
                     break
                 
            # 如果不是示例问题，添加到过滤后的列表
            if not is_example:
                 filtered_qa.append(item)
         
        # 确保至少有3个有效的QA对
        if len(filtered_qa) >= num_qa:
            # 只取前3个
            filtered_qa = filtered_qa[:num_qa]
             
            # 处理每个QA对
            for data in filtered_qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, filtered_qa
                # 处理答案格式
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"   
            return True, filtered_qa
    return False, qa

def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None, is_multi_doc=False, passage_list=None, num_qa=3):

    def fix_json(output):
        if"3.2-1b-instruct" in  model_name.lower():
            output = output[output.find("["):]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif "3-8b-instruct" in model_name.lower():
            if "[" in output:
                output = output[output.find("["):] 
            if "]" in output:
                output = output[:output.find("]")+1]
        # 确保输出以[]包裹（基础格式修正）
        output = output.strip()
        if not output.startswith("["):
            output = "[" + output
        if not output.endswith("]"):
            output += "]"

        # 提取数组内部的对象字符串
        inner = output[1:-1].strip()
        if not inner:  # 空数组处理
            return "[]"
        
        last_open = inner.rfind("{")
        last_close = inner.rfind("}", last_open)
        if last_close == -1:
            prev_close = inner.rfind("}", 0, last_open)
            inner = inner[:prev_close+1].strip() if prev_close != -1 else ""
        else:
            inner = inner[:last_close+1].strip()
        
        # 多余,
        inner = inner[:-1].strip() if inner.endswith(",") else inner

        return f"[{inner}]"

    try_times = 7
    # 根据是否为多文档选择不同的提示模板
    if is_multi_doc and passage_list:
        # 格式化多文档为带编号的形式
        formatted_passages = ""
        for i, p in enumerate(passage_list, 1):
            formatted_passages += f"Passage {i}: {p}\n\n"
        print(f"formatted_passages: {formatted_passages}")
        prompt = trans_document_qa_prompt_template.format(passage=formatted_passages)
    else:
        prompt = qa_prompt_template.format(passage=passage)
    
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        if num_qa == 2:
            print(f"passages:{passage},output: {output}")
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa, num_qa=num_qa)
            print(f"ret：{ret},qa:{qa}")
            if ret:
                return qa
        except:
            try_times -= 1
    return output
    

def main(args):
    output_dir = os.path.join(ROOT_DIR, args.output_dir, args.dataset, args.model_name.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path, args.dataset, args.model_name)

    if len(load_dataset) == 1:
        solve_dataset = load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(load_dataset["total"], fout, indent=4)
    
    model, tokenizer, _ = get_model(args.model_name)
    args.model_name = args.model_name.split("/")[-1]
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )
    popqa_error_list = []
    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        output_file = os.path.join(
            output_dir, 
            filename if filename.endswith(".json") else filename + ".json"
        )
        # print(output_file)
        # ret = []
        cnt=0
        error_id=[]
        pbar = tqdm(total = len(dataset))
        for data in dataset:
            cnt+=1
            # if args.dataset == "2wikimultihopqa" and cnt not in error_id:
            #     continue
            print(f"question: {data['question']}")
            if 'passages' not in data or len(data['passages']) == 0:
                passages = bm25_retrieve(data["question"], topk=args.topk+10)
            else:
                passages = data['passages']
            # print(f"len(passages): {len(passages)}")
            final_passages = []
            data["augment"] = []
            single_doc_augments = []  # 存储单文档增强结果，用于后续组合
            
            # 1. 处理单文档增强
            for psg in passages:
                val = { 
                    "pid": len(data["augment"]), 
                    "passage": psg, 
                    "is_multi_doc":False,
                    f"{args.model_name}_rewrite": get_rewrite(psg, args.model_name, model, tokenizer, generation_config)
                }
                # print(f"{args.model_name}_rewrite: {val[f'{args.model_name}_rewrite']}")
                qa = get_qa(psg, args.model_name, model, tokenizer, generation_config)
                if fix_qa(qa)[0] == False: # skip error passage
                    error_id.append(data["test_id"])
                    print(f"{filename}:{cnt}")
                    continue
                
                val[f"{args.model_name}_qa"] = qa
                data["augment"].append(val)
                single_doc_augments.append(val)  # 保存单文档增强结果
                final_passages.append(psg)
                pbar.update(1)
                if len(data["augment"]) == args.topk:
                    break
            
            # 2. 处理文档组合增强（如果有至少2个文档）
            if args.enable_multi_doc and len(single_doc_augments) >= 2:
                # 两两组合
                for i in range(len(single_doc_augments)):
                    for j in range(i + 1, len(single_doc_augments)):
                        psg1 = single_doc_augments[i]
                        psg2 = single_doc_augments[j]
                        
                        # 拼接原始文档和重写文档
                        combined_passage = psg1["passage"] + " [sep] " + psg2["passage"]
                        combined_rewrite = psg1[f"{args.model_name}_rewrite"] + " [sep] " + psg2[f"{args.model_name}_rewrite"]
                        
                        all_passages = [aug["passage"] for aug in single_doc_augments]
                        all_rewrites = [aug[f"{args.model_name}_rewrite"] for aug in single_doc_augments]
                        # 生成跨文档QA
                        passage_list = [psg1["passage"], psg2["passage"]]
                        # print(f"len(psg):{len(passage_list)}")
                        cross_doc_qa = get_qa(combined_passage, args.model_name, model, tokenizer, generation_config, 
                                            is_multi_doc=True, passage_list=passage_list, num_qa=2)
                        
                        # 构建组合文档的增强数据，第一个QA使用psg1的第一个QA，第二个QA使用跨文档生成的QA
                        combined_qa = []
                        if psg1[f"{args.model_name}_qa"]:
                            combined_qa.append(psg1[f"{args.model_name}_qa"][0])  # 添加第一个文档的第一个QA
                        if cross_doc_qa:
                            combined_qa.append(cross_doc_qa[0])  # 添加跨文档生成的QA
                        
                        # 确保至少有两个QA
                        while len(combined_qa) < 3 and cross_doc_qa and len(cross_doc_qa) > len(combined_qa) - 1:
                            combined_qa.append(cross_doc_qa[len(combined_qa) - 1])
                        
                        val = {
                            "pid": len(data["augment"]),
                            "passage": combined_passage,
                            "is_multi_doc": True,  # ← 标记多文档
                            f"{args.model_name}_rewrite": combined_rewrite,
                            f"{args.model_name}_qa": combined_qa
                        }
                        data["augment"].append(val)
                
            
            data["passages"] = final_passages
            # ret.append(data)
            # import pdb; pdb.set_trace()
            with open(output_file, "a") as fout:
                json.dump(data, fout, indent=4)
                fout.write(",\n")
        
        print(f"error_id: {error_id}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./llama-3.2-1b-instruct", )
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa", )
    parser.add_argument("--data_path", type=str, default="./data/2wikimultihopqa/", )
    parser.add_argument("--sample", type=int, default=300, )
    parser.add_argument("--topk", type=int, default=3) 
    parser.add_argument("--output_dir", type=str, default="data_aug")
    parser.add_argument("--projector", action="store_true")
    # 添加参数控制是否启用多文档组合增强
    parser.add_argument("--enable_multi_doc", action="store_true", default=True, 
                      help="Enable multi-document combination augmentation")
    args = parser.parse_args()
    print(args)
    main(args)