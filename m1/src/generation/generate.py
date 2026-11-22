import argparse
import json
import os
import math
import multiprocessing
import random
import numpy as np
from typing import List, Dict, Union, Any

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def worker_inference(
    worker_rank: int,
    gpu_ids: List[int], 
    prompts: List[Union[str, List[Dict]]], 
    model_name: str, 
    max_tokens: int, 
    max_model_len: int,
    mode: str,  # 'generation' or 'continuation'
    chat_template_args: Dict[str, Any],
    tensor_parallel_size: int,
    queue: multiprocessing.Queue,
    debug_mode: bool = False,
    output_dir: str = ""
):


    """
    Independent worker process that runs on a single GPU.
    """
    try:
        # 1. Isolate this process to specific GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        worker_id = gpu_ids[0]  # Physical GPU ID for logging
        
        # --- LOGGING CONFIGURATION ---
        # Set environment variables for C++ based logging and vLLM internal logging
        # Setting to WARNING ensures we see potential issues but hide INFO noise
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" # 1 = Filter INFO, show WARNING/ERROR
        
        # Configure Python logging
        import logging
        
        # Configure specific loggers to suppress INFO messages
        # This handles the "Graph capturing finished", "init engine", and "flashinfer" logs
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("cuda").setLevel(logging.WARNING)
        logging.getLogger("autotuner").setLevel(logging.WARNING)

        # 3. Import vLLM
        from vllm import LLM, SamplingParams

        print(f"[Worker {worker_id}] Initializing model on GPUs {gpu_ids} (TP={tensor_parallel_size})...")
        sampling_max_tokens = max_tokens if max_tokens != -1 else 16384

        if sampling_max_tokens<2000:
            max_num_seqs = 256
        else:
            max_num_seqs = 64
        # 4. Initialize Engine
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            # local_files_only=True
        )
        
        # --- PROMPT PREPARATION LOGIC ---
        
        print(f"[Worker {worker_id}] Mode: {mode.upper()}")
        
        tokenizer = llm.get_tokenizer()
        formatted_prompts = []
        
        for p in prompts:
            # STRICT INPUT HANDLING
            if mode == 'generation':
                if not isinstance(p, str):
                    raise ValueError(f"[Worker {worker_id}] Error: In 'generation' mode, input must be a string. Got {type(p)}.")
                messages = [{"role": "user", "content": p}]
                
                # Standard generation: formatting safe user input
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    **chat_template_args 
                )
                
            elif mode == 'continuation':
                if not isinstance(p, list):
                    raise ValueError(f"[Worker {worker_id}] Error: In 'continuation' mode, input must be a list of messages. Got {type(p)}.")
                
                if not p:
                     raise ValueError(f"[Worker {worker_id}] Error: Continuation input list cannot be empty.")

                # SPLIT STRATEGY: 
                # 1. Format history (safe) to generate headers/system prompt
                # 2. Manually append partial content (potentially unsafe/raw)
                
                history = p[:-1]
                last_msg = p[-1]
                
                # Step 1: Generate prefix up to "Assistant:"
                prefix = tokenizer.apply_chat_template(
                    history, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    **chat_template_args 
                )
                
                # Step 2: Append the raw partial content
                text = prefix + str(last_msg['content'])
            
            else:
                 raise ValueError(f"[Worker {worker_id}] Error: Unknown mode '{mode}'")

            formatted_prompts.append(text)
        
        final_prompts = formatted_prompts
        # ---------------------------

        # Detect QwQ model and use optimized settings
        if "qwq" in model_name.lower():
            print(f"[Worker {worker_id}] Detected QwQ model, using recommended settings: t=0.6, top_p=0.95, top_k=20")
            temperature = 0.6
            top_p = 0.95
            top_k = 20
        else:
            temperature = 0.7
            top_p = 0.9
            top_k = -1

        # Handle infinite generation
        sampling_params = SamplingParams(
            max_tokens=sampling_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_special_tokens=False,
        ) 



        print(f"[Worker {worker_id}] Processing batch of {len(final_prompts)} prompts...")
        
        # Debug logging moved to after generation to include outputs and token IDs


        # 5. Generate

        outputs = llm.generate(final_prompts, sampling_params, use_tqdm=True)
        
        # 6. Extract text
        generated_texts = [output.outputs[0].text for output in outputs]

        if debug_mode and output_dir:
             debug_file = os.path.join(output_dir, f"worker_{worker_id}_debug.log")
             with open(debug_file, "w", encoding="utf-8") as f: # Overwrite existing or create new
                 f.write(f"Worker {worker_id} Debug Log\n")
                 f.write("="*40 + "\n\n")
                 
                 for i, output in enumerate(outputs):
                     prompt = output.prompt
                     generated_text = output.outputs[0].text
                     token_ids = output.outputs[0].token_ids
                     
                     f.write(f"--- Example {i} ---\n")
                     f.write(f"PROMPT:\n{prompt}\n")
                     f.write("-" * 20 + "\n")
                     f.write(f"OUTPUT:\n{generated_text}\n")
                     f.write("-" * 20 + "\n")
                     f.write(f"TOKEN IDS:\n{token_ids}\n")
                     f.write("=" * 40 + "\n\n")


        
        # 7. Send back results
        queue.put((worker_rank, generated_texts))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put((worker_rank, e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory. Defaults to the same directory as --input_file if not provided.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=-1, help="Number of tokens to continue generating. -1 for infinite.")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum context length for the model.")
    parser.add_argument("--num_gpus", type=int, default=8, help="Total number of GPUs")
    parser.add_argument("--data_parallel", type=int, default=4, help="Number of data parallel workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--mode", type=str, choices=['generation', 'continuation'], default='continuation',
                        help="Mode 'generation' uses 'prompt' field. 'continuation' uses 'conversations' field.")
    
    parser.add_argument("--chat_template_args", type=str, default="{}", 
                        help="JSON string of extra args for chat template")
    
    parser.add_argument("--continuation_indices", type=str, default=None,
                        help="Path to .npy file containing indices to continue (required for continuation mode)")
    
    parser.add_argument("--subset_indices", type=str, default=None,
                        help="Path to .npy file containing indices to generate for (generation mode only). "
                             "Only the rows at these indices will be processed; all others are skipped.")
    
    parser.add_argument("--box_prompt", type=str, choices=['begin', 'end', 'none'], default='none',
                        help="Add boxed prompt instruction to user prompt: 'begin' (prepend), 'end' (append), or 'none'")
    
    parser.add_argument("--answer_type", type=str, choices=['choice', 'others'], default='others',
                        help="Type of answer expected: 'choice' (multiple choice letter) or 'others' (general response)")
    
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="Optional generation-mode template. Use {question} or {prompt} for the row prompt/problem text.")
    
    args = parser.parse_args()
    
    # Default output_dir to input_file's directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_file)
        if not args.output_dir:  # If input_file has no directory component
            args.output_dir = "."
        print(f"No output_dir specified. Using input_file directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output dir exists for debug logs
    
    # Compute output filename and check for overwrite
    output_filename = f"{args.mode}_results.jsonl"
    if args.debug:
        output_filename = f"debug_{output_filename}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    if os.path.exists(output_path):
        raise FileExistsError(f"Output file already exists and would be overwritten: {output_path}")



    # Validate continuation_indices requirement for continuation mode
    if args.mode == 'continuation':
        if args.continuation_indices is None:
            raise ValueError("--continuation_indices is required when mode is 'continuation'")
        if not args.continuation_indices.endswith('.npy'):
            raise ValueError("--continuation_indices must be a .npy file path")
        if not os.path.exists(args.continuation_indices):
            raise FileNotFoundError(f"Continuation indices file not found: {args.continuation_indices}")

    try:
        chat_template_extra = json.loads(args.chat_template_args)
    except json.JSONDecodeError:
        chat_template_extra = {}

    multiprocessing.set_start_method('spawn', force=True)

    # 1. Load Data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)

    # if args.debug:
    #     print("Debug mode: processing first 100 examples only.")
    #     data = data[:100]

    total_items = len(data)
    
    # Load continuation indices if in continuation mode
    continuation_indices_set = None
    if args.mode == 'continuation':
        print(f"Loading continuation indices from {args.continuation_indices}...")
        continuation_indices = np.load(args.continuation_indices)
        continuation_indices_set = set(continuation_indices.tolist())
        print(f"Loaded {len(continuation_indices_set)} indices to continue")
        
        # Validate indices
        invalid_indices = [idx for idx in continuation_indices_set if idx >= total_items or idx < 0]
        if invalid_indices:
            raise ValueError(f"Invalid indices found (out of range [0, {total_items})): {invalid_indices[:10]}...")
    
    # Selection logic based on mode with fallback
    print(f"Extracting input text (Mode: {args.mode})...")

    raw_prompts = []
    
    # Pre-define boxed prompt instruction
    if args.answer_type == 'choice':
        BOXED_INSTRUCTION = "Return only the single option letter within \\boxed{}."
    else:
        BOXED_INSTRUCTION = "Return your final response within \\boxed{}."

    for item in data:
        content = None
        using_fallback = False

        if args.mode == 'generation':
            if "prompt" in item:
                content = item["prompt"]
            elif "problem" in item:
                # Fallback to 'problem' field if 'prompt' is missing
                content = item["problem"]
            elif "conversations" in item:
                # Fallback to user input from conversations[0]
                if isinstance(item["conversations"], list) and len(item["conversations"]) > 0:
                    content = item["conversations"][0].get("content", "")
                    using_fallback = True
        elif args.mode == 'continuation':
             if "conversations" in item:
                 content = item["conversations"]
        
        if content is not None:
            if args.mode == 'generation' and args.prompt_template is not None:
                if not isinstance(content, str):
                    raise ValueError(f"Content for prompt_template modification is not a string. Type: {type(content)}.")
                content = args.prompt_template.format(question=content, prompt=content)

            # Apply box prompt modification ONLY in generation mode AND if NOT using fallback
            if args.mode == 'generation' and args.box_prompt != 'none' and not using_fallback:
                if isinstance(content, str):
                    if args.box_prompt == 'end':
                        # Ensure newline separator
                        if not content.endswith('\n'):
                            content += "\n"
                        content += BOXED_INSTRUCTION
                    elif args.box_prompt == 'begin':
                        # Prepend with space check
                        content = BOXED_INSTRUCTION + " " + content
                else:
                    raise ValueError(f"Content for box_prompt modification is not a string. Type: {type(content)}. Skipping modification.")   

            raw_prompts.append(content)
        else:
            target_desc = "prompt (or conversations[0])" if args.mode == 'generation' else "conversations"
            raise ValueError(f"Field '{target_desc}' not found in item {item.get('id', '?')}. Appending empty.")   

    # --- LOAD BALANCING ---
    # Filter prompts based on continuation indices if in continuation mode
    if args.mode == 'continuation':
        # Only process items that need continuation
        indices_to_process = [i for i in range(total_items) if i in continuation_indices_set]
        print(f"Processing {len(indices_to_process)} items out of {total_items} total items")
    elif args.subset_indices is not None:
        # Generation mode with explicit subset filter
        subset = np.load(args.subset_indices)
        subset_set = set(int(i) for i in subset)
        invalid = [i for i in subset_set if i < 0 or i >= total_items]
        if invalid:
            raise ValueError(f"--subset_indices: indices out of range [0, {total_items}): {invalid[:10]}")
        indices_to_process = [i for i in range(total_items) if i in subset_set]
        print(f"Processing {len(indices_to_process)} items (subset) out of {total_items} total items")
    else:
        # Process all items for generation mode
        indices_to_process = list(range(total_items))
    
    random.seed(args.seed)
    random.shuffle(indices_to_process)
    
    prompts_to_process = [raw_prompts[i] for i in indices_to_process]
    
    # Calculate tensor parallel size
    tensor_parallel_size = args.num_gpus // args.data_parallel
    if tensor_parallel_size < 1:
        tensor_parallel_size = 1
        
    print(f"Using {args.data_parallel} data parallel workers, each with TP={tensor_parallel_size}")
    
    # Parse CUDA_VISIBLE_DEVICES if present to get physical device mapping
    # logical_id -> physical_id
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        physical_devices = [int(x) for x in cuda_visible.split(",") if x.strip()]
        if len(physical_devices) < args.num_gpus:
             # This might happen if user claimed more GPUs than visible, vLLM would fail anyway
             print(f"WARNING: num_gpus ({args.num_gpus}) > visible devices ({len(physical_devices)}). Mapping might fail.")
    else:
        # Assume 0..N are the physical IDs
        physical_devices = list(range(args.num_gpus))
    
    chunk_size = math.ceil(len(prompts_to_process) / args.data_parallel)
    chunks = []
    for i in range(args.data_parallel):
        start = i * chunk_size
        end = min(start + chunk_size, len(prompts_to_process))
        if start < len(prompts_to_process):
            chunks.append(prompts_to_process[start:end])
        else:
            chunks.append([])

    # 3. Launch Workers
    queue = multiprocessing.Queue()
    processes = []
    
    print(f"Distributing work (Mode: {args.mode})...")

    for i in range(args.data_parallel):
        if not chunks[i]:
            continue
        
        # Logical GPU IDs for this worker (0-based relative to visible)
        logical_gpu_ids = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        
        # Map to physical IDs
        physical_gpu_ids = []
        for lid in logical_gpu_ids:
            if lid < len(physical_devices):
                physical_gpu_ids.append(physical_devices[lid])
            else:
                raise ValueError(f"Logical GPU ID {lid} out of range for visible devices {physical_devices}")
            
        p = multiprocessing.Process(
            target=worker_inference,
            args=(
                i,
                physical_gpu_ids, 
                chunks[i], 
                args.model_name, 
                args.max_tokens, 
                args.max_model_len,
                args.mode,
                chat_template_extra,
                tensor_parallel_size,
                queue,
                args.debug,
                args.output_dir
            )


        )
        p.start()
        processes.append(p)

    # 4. Collect Results
    results_map = {}
    active_processes = len([c for c in chunks if c])
    
    for _ in range(active_processes):
        item = queue.get()
        if isinstance(item[1], Exception):
            for p in processes: p.terminate()
            raise item[1]
        results_map[item[0]] = item[1]

    for p in processes:
        p.join()

    # 5. Reassemble
    processed_results = []
    for i in range(args.data_parallel):
        if i in results_map:
            processed_results.extend(results_map[i])

    if len(processed_results) != len(indices_to_process):
        raise ValueError(f"CRITICAL ERROR: Mismatch in processed items! Expected: {len(indices_to_process)}, Output: {len(processed_results)}")

    # 6. Save Data
    print("Reassembling and saving...")
    
    # Load tokenizer for counting actual tokens when max_tokens == -1
    tokenizer = None
    if args.max_tokens == -1:
        from transformers import AutoTokenizer
        print("Loading tokenizer for counting generated tokens...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create a mapping from original index to generated result
    result_mapping = {}
    for k, original_idx in enumerate(indices_to_process):
        result_mapping[original_idx] = processed_results[k]
    
    # Process all items in original order
    for original_idx in range(total_items):
        original_item = data[original_idx]
        
        # Check if this item was processed
        if original_idx not in result_mapping:
            # Item was not processed - keep it unchanged
            continue
        
        # Item was processed - update it
        new_tokens = result_mapping[original_idx]
        input_text = raw_prompts[original_idx]
        
        conversations = []
        
        # Calculate final max_tokens
        original_max_tokens = original_item.get("max_tokens", 0)
        if not isinstance(original_max_tokens, int):
             original_max_tokens = 0
        
        # Calculate actual token count for generated text if using unlimited generation
        if args.max_tokens == -1:
            # Count actual tokens generated (only for processed items, saves time)
            generated_token_count = len(tokenizer.encode(new_tokens, add_special_tokens=False))
        else:
            generated_token_count = args.max_tokens

        if args.mode == 'continuation':
            # Accumulate tokens for continuation
            final_max_tokens = original_max_tokens + generated_token_count
            
            # STRICT REASSEMBLY FOR CONTINUATION
            if not isinstance(input_text, list):
                raise ValueError(f"Reassembly Error: Mode is 'continuation' but input was not a list. Type: {type(input_text)}")
            
            conversations = [dict(msg) for msg in input_text]
            
            if not conversations:
                raise ValueError("Reassembly Error: Continuation input list is empty.")

            last_msg = conversations[-1]
            if last_msg.get('role') != 'assistant':
                raise ValueError(f"Reassembly Error: In 'continuation' mode, the last message must be from 'assistant'. Found: {last_msg.get('role')}")

            # Append new tokens to the existing last message
            last_msg['content'] = str(last_msg.get('content', '')) + new_tokens
                
        elif args.mode == 'generation':
            # For generation, use actual generated count
            final_max_tokens = generated_token_count
            
            # STRICT REASSEMBLY FOR GENERATION
            if not isinstance(input_text, str):
                raise ValueError(f"Reassembly Error: Mode is 'generation' but input was not a string. Type: {type(input_text)}")
            
            conversations = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": new_tokens}
            ]

        # Construct final dict (verbatim + optional id)
        final_item = {
            "prompt": original_item.get("prompt") or original_item.get("problem") or conversations[0]['content'],  # Preserve original prompt/problem if present, else use input
            "source": original_item.get("source"),
            "conversations": conversations,
            "max_tokens": final_max_tokens
        }
        
        # PRESERVE ID IF PRESENT (Helps check order)
        if "id" in original_item:
            final_item["id"] = original_item["id"]

        for metadata_key in (
            "original_index",
            "Problem_ID",
            "domain",
            "dataset",
            "category",
            "problem_difficulty",
            "problem_difficulty_merged",
        ):
            if metadata_key in original_item:
                final_item[metadata_key] = original_item[metadata_key]
            
        data[original_idx] = final_item


    os.makedirs(args.output_dir, exist_ok=True)
    
    save_jsonl(data, output_path)
    print(f"Saved {len(data)} items to {output_path}, {len(indices_to_process)} items processed.")

if __name__ == "__main__":
    main()
