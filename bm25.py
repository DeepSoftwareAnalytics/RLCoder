from rank_bm25 import BM25Okapi
from multiprocessing import Pool
from collections import defaultdict
from typing import List
from datasets import CodeBlock
from functools import partial
import math

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def split_into_smaller_blocks(code_block, enable_fixed_block):
    """
    Split large blocks of code into smaller ones, each containing no more than 12 non-empty lines.
    """
    smaller_blocks = []

    if enable_fixed_block:
        lines = [line for line in code_block.code_content.split('\n') if line.strip() != '']
        for i in range(0, min(len(lines),5000), 12):
            start_line_offset = i
            end_line_offset = min(i + 12, len(lines))
            block_content = '\n'.join(lines[start_line_offset:end_line_offset])
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            f"file path: {code_block.file_path}\nlines: {start_line_offset}-{end_line_offset - 1}",
                                            block_content,
                                            code_block.language,
                                            'fixed_block'))

    else:
        # Split the code by spaces, then reassemble it into blocks.
        mini_blocks = []
        current_block = [] 
        for line in code_block.code_content.splitlines(): 
            if line.strip() == '':  
                if current_block: 
                    mini_blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)
        if current_block: 
            mini_blocks.append(current_block)

        max_len = 15
        temp_mini_blocks = []
        for mini_block in mini_blocks:
            if len(mini_block) > max_len:
                for idx in range(0, len(mini_block), max_len):
                    temp_mini_blocks.append(mini_block[idx: idx+max_len])
            else:
                temp_mini_blocks.append(mini_block)
        mini_blocks = temp_mini_blocks

        current_content = []
        total_lines = 0  
        for block in mini_blocks:
            if total_lines >= 5000:  
                break  
            if len(current_content) + len(block) <= 15:  
                current_content.extend(block)
                total_lines += len(block)  
            else:  
                if current_content:  
                    smaller_blocks.append(CodeBlock(code_block.file_path, 
                                                    f"file path: {code_block.file_path}\nlines: {total_lines-len(current_content)+1}-{total_lines}",
                                                    '\n'.join(current_content),
                                                    code_block.language,
                                                    'mini_block'))
                current_content = block  
                total_lines += len(block)  
        if current_content:  
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            f"file path: {code_block.file_path}\nlines: {total_lines-len(current_content)+1}-{total_lines}",
                                            '\n'.join(current_content),
                                            code_block.language,
                                            'mini_block'))
        
    return smaller_blocks

class TaskSpecificBM25:
    def __init__(self, examples, args):
        self.bm25_indices = {}
        self.code_blocks = {}
        self._build_indices(examples, args)
        self.args = args
        
    def _build_indices(self, examples, args):
        num_processes = 32  
        num_examples_per_batch = math.ceil(len(examples) / num_processes)

    
        example_batches = [examples[i:i + num_examples_per_batch] for i in range(0, len(examples), num_examples_per_batch)]

        with Pool(processes=num_processes) as pool:
            # results = pool.map(self._process_batch, example_batches, args.enable_fixed_block)
            # results = pool.map(lambda batch: self._process_batch(batch, args.enable_fixed_block), example_batches)
            from functools import partial
            partial_process_batch = partial(self._process_batch, enable_fixed_block=args.enable_fixed_block)
            results = pool.map(partial_process_batch, example_batches)
        
        for batch_result in results:
            for task_id, code_blocks, bm25_index in batch_result:
                self.bm25_indices[task_id] = bm25_index
                self.code_blocks[task_id] = code_blocks

        block_len = 0
        block_num = 0
        for task_id, code_blocks in self.code_blocks.items():
            block_num += len(code_blocks)
            for code_block in code_blocks:
                block_len += len(code_block.code_content.splitlines())
        
        logger.info(f'Block avg line: {round(block_len / block_num, 2)}')

    @staticmethod
    def _process_batch(batch, enable_fixed_block):
        batch_result = []
        for example in batch:
            code_blocks = []
            for code_block in example.related_files:
                code_blocks.extend(split_into_smaller_blocks(code_block, enable_fixed_block))
            
            bm25_index = None
            if len(code_blocks) != 0:
                bm25_index = BM25Okapi([code_block.code_content.lower().split() for code_block in code_blocks])
            
            batch_result.append((example.task_id, code_blocks, bm25_index))
        return batch_result

    def query(self, task_ids: List[int], queries: List[str], topk: int):
        results = []
        for task_id, query in zip(task_ids, queries):
            bm25_index = self.bm25_indices.get(task_id)
            if bm25_index:
                query_tokens = query.split()
                scores = bm25_index.get_scores(query_tokens)

                # topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
                # task_results = [self.code_blocks[task_id][i] for i in topk_indices]
                # results.append(task_results)

                #  ||
                # \||/
                #  \/

                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                fixed_block_indices = []
                signature_block_indices = []
                full_func_block_indices = []
                mini_block_indices = []
                for i in sorted_indices:
                    block = self.code_blocks[task_id][i]
                    if block._type == "fixed_block":
                        fixed_block_indices.append(i)
                    elif block._type == "signature_block":
                        signature_block_indices.append(i)
                    elif block._type == "full_func_block":
                        full_func_block_indices.append(i)
                    elif block._type == "mini_block":
                        mini_block_indices.append(i)
                    
                topk_fixed_blocks = [self.code_blocks[task_id][i] for i in fixed_block_indices[:topk]]
                # topk_signature_blocks = [self.code_blocks[task_id][i] for i in signature_block_indices[:topk]]
                # topk_full_func_blocks = [self.code_blocks[task_id][i] for i in full_func_block_indices[:topk]]
                topk_mini_blocks = [self.code_blocks[task_id][i] for i in mini_block_indices[:topk]]

                if self.args.enable_fixed_block:
                    task_results = topk_fixed_blocks
                else:
                    task_results = topk_mini_blocks
                results.append(task_results)
            else:
                results.append([])
            #results.append([CodeBlock("__init__.py", 0, 0, "") for _ in range(topk+1)])
        return results