import json
import re
import subprocess
import argparse
from pprint import pprint

def count_indent(line):
    count = 0
    for char in line:
        if char == ' ':
            count += 1
        elif char == '\t':
            count += 4
        else:
            break
    return count

def extract_func(code, language='python'):
    if language == 'python':
        match = re.search(r'(^def[\s\S]*?\n[\s\S]*?)\n[#@a-zA-Z]', code)
        if match:
            first_function = match.groups()[0]
        else:
            match = re.search(r'(^def[\s\S]*?\n[\s\S]*?)\n(def|class) ', code)
            if match:
                first_function = match.groups()[0]
            else:
                match = re.search(r'(^def[\s\S]*?\n[\s\S]*?)if __name__', code)
                if match:
                    first_function = match.groups()[0]
                else:
                    first_function = code

        first_function = first_function.strip()
        
    elif language == 'java':
        end_idx = 0
        cnt = 0
        flag = 0
        for token in code:
            end_idx += 1
            if token == '{':
                cnt += 1
                flag = 1
            elif token == '}':
                cnt -= 1
            if flag and (cnt == 0):
                break
        
        first_function = code[:end_idx]
        
    return first_function

def postProcess(prompt_lines, pred_lines, output_dir, language='python'):
    generate_result_list = []

    for i in range(len(pred_lines)):
        pred_line = pred_lines[i]
        _id = pred_line['task_id']
        pred = pred_line['pred']

        for prompt_line in prompt_lines:
            if prompt_line['question_id'] == _id:
                signature = prompt_line['signature']
                break
        
        generate_results = []
        generate_result = signature + '\n'
        
        first_flag = True
        del_indent = 0
        for line in pred.splitlines():
            if line.strip() != '':
                if first_flag:
                    first_flag = False
                    del_indent = count_indent(line) - 4
                    generate_result += ' ' * 4 + line.strip() + '\n'
                else:
                    generate_result += line[del_indent:] + '\n'
            else:
                generate_result += '\n'
        
        generate_result = extract_func(generate_result, language)
        # print(generate_result)
        # input()     
        generate_results = [generate_result] * 10
        
        generate_result = {
            '_id': _id,
            'generate_results': generate_results,
        }
        generate_result_list.append(generate_result)

    pred_truncated_file = 'prediction_truncated.jsonl'
    
    with open(f'{output_dir}/{pred_truncated_file}', 'w', encoding='utf-8') as f:
        for generate_result in generate_result_list:
            json.dump(generate_result, f)
            f.write('\n')

def eval_in_docker(output_dir, language='python', container_name=None):
    if container_name is None:
        container_name = ''

    pred_truncated_file = 'prediction_truncated.jsonl'
    results = {
            'count': '-',
            'all': '-',
            'self': '-',
            'slib': '-',
            'plib': '-',
            'class': '-',
            'file': '-',
            'project': '-',
        }

    if language == 'python':
        root_dir = '' # codereval eval path
        
        subprocess.run(['docker', 'exec', container_name, 'mkdir', '-p', f'{root_dir}/{language}_pred'])
        subprocess.run(['docker', 'cp', f'{output_dir}/{pred_truncated_file}', f'{container_name}:{root_dir}/{language}_pred'])
        subprocess.run(['docker', 'exec', container_name, 'python', f'{root_dir}/GroundTruth.py'], capture_output=True, text=True)
        eval_result = subprocess.run(['docker', 'exec', container_name, 'python', f'{root_dir}/PythonExec.py', f'{root_dir}/{language}_pred/{pred_truncated_file}', '10'], capture_output=True, text=True)
        eval_result_text = eval_result.stdout
        eval_result_text_list = eval_result_text.splitlines()
        
        for i in range(len(eval_result_text_list)):
            if eval_result_text_list[i].strip() == 'finish_overall':
                cnt = int(float(eval_result_text_list[i-21].strip().split(', ')[1]))
                all = float(eval_result_text_list[i-1].strip().strip('%'))

                results = {
                    'count': cnt,
                    'all': all,
                    'self': float(eval_result_text_list[i+45].strip().strip('%')),
                    'slib': float(eval_result_text_list[i+73].strip().strip('%')),
                    'plib': float(eval_result_text_list[i+3].strip().strip('%')),
                    'class': float(eval_result_text_list[i+31].strip().strip('%')),
                    'file': float(eval_result_text_list[i+59].strip().strip('%')),
                    'project': float(eval_result_text_list[i+17].strip().strip('%')),
                }
                break
        
        return results, eval_result_text
        
    elif language == 'java':
        root_dir = '' # codereval eval path for java
        
        subprocess.run(['docker', 'exec', container_name, 'mkdir', '-p', f'{root_dir}/{language}_pred'])
        subprocess.run(['docker', 'cp', f'{output_dir}/{pred_truncated_file}', f'{container_name}:{root_dir}/{language}_pred'])
        subprocess.run(['docker', 'exec', container_name, 'python', f'{root_dir}/GroundTruth.py'], capture_output=True, text=True)
        eval_result = subprocess.run(['docker', 'exec', container_name, 'python', f'{root_dir}/JavaExec.py', f'{root_dir}/{language}_pred/{pred_truncated_file}', '10'], capture_output=True, text=True)
        eval_result_text = eval_result.stdout
        print(eval_result.stderr)

        eval_result_text_list = eval_result_text.splitlines()
        
        for i in range(len(eval_result_text_list)):
            if eval_result_text_list[i].strip() == 'finish_overall':
                cnt = int(float(eval_result_text_list[i-21].strip().split(', ')[1]))
                all = float(eval_result_text_list[i-1].strip().strip('%'))

                results = {
                    'count': cnt,
                    'all': all,
                    'self': float(eval_result_text_list[i+3].strip().strip('%')),
                    'slib': float(eval_result_text_list[i+17].strip().strip('%')),
                    'plib': float(eval_result_text_list[i+59].strip().strip('%')),
                    'class': float(eval_result_text_list[i+31].strip().strip('%')),
                    'file': float(eval_result_text_list[i+73].strip().strip('%')),
                    'project': float(eval_result_text_list[i+45].strip().strip('%')),
                }
                break
        
        return results, eval_result_text


def eval_codereval(output_dir, prompt_file, language='python', do_codereval=False):
    with open(prompt_file, encoding='utf-8') as f:
        prompt_lines = f.readlines()
    prompt_lines = [json.loads(prompt_line) for prompt_line in prompt_lines]

    pred_file = 'prediction.jsonl'
        
    with open(f'{output_dir}/{pred_file}', encoding='utf-8') as f:
        pred_lines = f.readlines()
    pred_lines = [json.loads(pred_line) for pred_line in pred_lines]

    postProcess(prompt_lines, pred_lines, output_dir, language)
    
    if do_codereval:
        results, eval_result_text = eval_in_docker(output_dir, language)

        with open(f'{output_dir}/eval_result_text.txt', 'w', encoding='utf-8') as f:
            f.write(eval_result_text)
        with open(f'{output_dir}/eval_result.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results
    else:
        return  {
            'count': '-',
            'all': '-',
            'self': '-',
            'slib': '-',
            'plib': '-',
            'class': '-',
            'file': '-',
            'project': '-',
        }

if __name__ == '__main__':
    # eval_codereval('results/tmp/result_0/codereval_python', 'data/codereval/python/CEPythonRaw.jsonl', language="python", do_codereval=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="", type=str, help="prediction file path")
    parser.add_argument("--language", default="python", type=str, help="language")
    parser.add_argument("--container_name", default=None, type=str, help="docker name")
    args = parser.parse_args()
    results, eval_result_text = eval_in_docker(args.output_dir, args.language, args.container_name)
    
    with open(f'{args.output_dir}/eval_result_text.txt', 'w', encoding='utf-8') as f:
        f.write(eval_result_text)
    with open(f'{args.output_dir}/eval_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(args.output_dir)
    pprint(results)
