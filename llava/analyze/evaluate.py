import torch
import os
import json
import argparse
import numpy as np
from utils import model_utils,utils

def parse_args():
    parser = argparse.ArgumentParser(description='VLGuard Evaluation')

    parser.add_argument('--metaDir', default='playground/data/vlguard/test.json', type=str)
    parser.add_argument('--imageDir', default='playground/data/vlguard/test', type=str)
    parser.add_argument('--dataset', default='safe_safes', type=str, choices=['safe_unsafes', 'safe_safes', 'unsafes'])

    #parser.add_argument("--engine", "-e", default=["llava15-7b"], nargs="+")

    parser.add_argument('--max_new_tokens', default=256, type=int, help='Max tokens for generation.')
    parser.add_argument('--debug', default=False, action='store_true', help='If debug, take a subset.')

    return parser.parse_args()

def load_data(args):
    subset = args.dataset
    
    with open(args.metaDir, 'r') as f:
        all_questions = json.load(f)
    if args.dataset == 'unsafes':
        all_questions = [q for q in all_questions if q['safe'] == False]
        for question in all_questions:
            question['question'] = question['instr-resp'][0]['instruction']
    else:
        all_questions = [q for q in all_questions if q['safe'] == True]
        key = 'unsafe_instruction' if subset == 'safe_unsafes' else 'safe_instruction'
        for question in all_questions:
            question['question'] = next(
                (q[key] for q in question['instr-resp'] if key in q), None
            )
    return all_questions


def eval_questions(args, questions, model, tokenizer, processor, root_path, engine, output_dir):

    results = {}
    total_succ = []

    with torch.no_grad():
        for question in questions:
            img_id = question['image']
            image, image_path = utils.load_image(img_id, root_path)
            question_text = question['question']
            
            predicted_answers = model_utils.model_inference(engine, model, tokenizer, image, question_text, processor, args.max_new_tokens, output_dir)

            break
            #results[question['image']] = predicted_answers

            #succ = utils.check_success_attack(predicted_answers)
            #total_succ.append(succ)

    # if args.dataset != 'safe_safes':
    #     succ_rate = round(np.mean(total_succ) * 100, 2)
    #     print(f'{args.dataset} ASR of {engine}: ', f"{succ_rate}", flush=True)

    return results

if __name__ == "__main__":
    args = parse_args()

    all_questions_safe = load_data(args)

    args.dataset = 'unsafes'
    all_questions_unsafe = load_data(args)
    model_path = "./llava-v1.5-7b"
    model, tokenizer, processor = model_utils.load_model(model_path, args)
    print("Loaded model: {}\n".format(model_path))

    out1="tsne_outputs_safe"
    out2="tsne_outputs_unsafe"
    results_dict = eval_questions(args, all_questions_safe, model, tokenizer, processor, args.imageDir, model_path, out1)
    
    results_dict = eval_questions(args, all_questions_unsafe, model, tokenizer, processor, args.imageDir, model_path, out2)
    # os.makedirs(f'results/{args.dataset}', exist_ok=True)
    # with open(f'results/{args.dataset}/{engine}.json', 'w') as f:
    #     json.dump(results_dict, f, indent=4)
    
    del model, tokenizer, processor
    torch.cuda.empty_cache()
        