import boto3
import pandas as pd
import re
import asyncio
import random
import wandb
import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from tqdm.asyncio import tqdm
from botocore.exceptions import ClientError
import verl.utils.torch_functional as verl_F

session = boto3.session.Session()
client = session.client('bedrock-runtime', region_name='us-west-2')

async def get_hint(failed_response_text, ground_truth, model_id, config):
    """Call LLM with exponential backoff retry logic."""
    teacher_system_prompt = "The user will give you a math question and his attempt to solve it." \
    " You are a math teacher. Your task is to provide constructive feedback on the user's attempt, " \
    "pointing out any mistakes or misconceptions, and guide them towards the correct solution." \
    "Please give the user a hint. Do not provide the full solution directly.  Be succinct." 
    
    # teacher_system_prompt = "The user will give you a question and their answer. Please succinctly tell the student that they are wrong and give them the correct answer."
    # Create the prompt with the failed response and ground truth
    prompt = f"Student's attempt:\n{failed_response_text}\n\nCorrect answer: {ground_truth}"
    
    teacher_prompt = teacher_system_prompt + prompt
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    msgs = [{'role': 'user', 'content': [{'text': teacher_prompt}]}]
    
    for attempt in range(config.get('max_retries', 3)):
        try:
            if config.get('verbose_logging', False):
                print(f"  → API call to {model_id.split('.')[-1]} (attempt {attempt+1})")
            
            res = await loop.run_in_executor(executor, lambda: client.converse(
                modelId=model_id, 
                messages=msgs, 
                system=[], 
                inferenceConfig={
                    'temperature': config.get('temperature', 0.7),
                    'maxTokens': config.get('max_tokens', 1024)
                }
            ))
            
            content = res['output']['message']['content']
            thinking, text = "", ""
            for item in content:
                if isinstance(item, dict):
                    if 'reasoningContent' in item:
                        thinking = item['reasoningContent'].get('reasoningText', {}).get('text', '')
                    elif 'text' in item:
                        text = item['text']
            
            if config.get('verbose_logging', False):
                print(f"  ✓ Response received")
            
            return text, thinking
            # return text + f"The correct answer is {ground_truth}." + " Please give the correct answer now inside of \boxed{}.", thinking
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                if attempt < config.get('max_retries', 3) - 1:
                    wait_time = (config.get('backoff_base', 2) ** attempt) + random.uniform(0, config.get('backoff_jitter_max', 1))
                    print(f"  ⚠ Rate limited (attempt {attempt+1}/{config.get('max_retries', 3)}). Waiting {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  ✗ Max retries exceeded due to rate limiting")
                    raise
            else:
                raise
        
        except Exception as e:
            print(f"  ✗ Unexpected error in call_llm: {e}")
            raise
    
    raise Exception(f"Max retries exceeded for model {model_id}")

def process_single_retry_item(
    i,
    unsuccessful_batch,
    tokenizer,
    ground_truth,
    config,
):
    # Get the conversation - it might be a numpy array or a list
    text_prompt_item = unsuccessful_batch.non_tensor_batch["text_prompt"][i]
    # Convert to list if it's a numpy array, then deep copy
    if isinstance(text_prompt_item, np.ndarray):
        conversation = copy.deepcopy(text_prompt_item.tolist())
    else:
        conversation = copy.deepcopy(text_prompt_item)

    # Get the failed response for THIS specific trace
    failed_response_ids = unsuccessful_batch.batch["responses"][i]
    failed_response_mask = unsuccessful_batch.batch["response_mask"][i]

    # Decode the failed response
    valid_response_mask = failed_response_mask.bool()
    valid_response_ids = failed_response_ids[valid_response_mask]
    failed_response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    failed_response_text = re.sub(
        r'^(assistant|user|system)\s*\n*',
        '',
        failed_response_text,
        flags=re.IGNORECASE
    ).strip()

    # Append the failed response to THIS conversation copy
    conversation.append({"content": failed_response_text, "role": "assistant"})

    # Append the retry message
    if config.trainer.get("teacher_model", False):
        # have a stronger model give a hint.
        model_id = config.trainer.get("teacher_model_id", "openai.gpt-oss-120b-1:0")
        # Note: get_hint is async, so this needs to be called with await in an async context
        # or use asyncio.run() if this function is not async
        hint_text, _ = asyncio.run(get_hint(failed_response_text, ground_truth, model_id, config))
        retry_message = {
            "role": "user",
            "content": hint_text
        }
    else:
        retry_message = {
            "role": "user",
            "content": (
                "Your previous answer was incorrect and/or you formatted it incorrectly. "
                "First, find your reasoning error and/or formatting error and explain why "
                "your answer was incorrect.  Please correct your reasoning and provide a new "
                "answer. Let's think step by step and do not forget to output the final answer "
                "inside \\boxed{}."
            ),
        }

    conversation.append(retry_message)

    # Apply chat template to this conversation
    retry_message_in_template = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        **config.data.get("apply_chat_template_kwargs", {})
    )

    # Tokenize
    revised_trace = tokenizer(
        retry_message_in_template,
        return_tensors="pt",
        add_special_tokens=False
    )
    revised_attention_mask = revised_trace.pop("attention_mask")
    revised_input_ids = revised_trace.pop("input_ids")

    # Postprocess
    try:
        revised_input_ids, revised_attention_mask = verl_F.postprocess_data(
            input_ids=revised_input_ids,
            attention_mask=revised_attention_mask,
            max_length=config.data.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
    except Exception:
        print(
            "The too-long response is:",
            tokenizer.decode(
                revised_input_ids[0][revised_attention_mask[0].bool()],
                skip_special_tokens=True
            )
        )

    # Return everything we need, including index so we can restore order
    return i, conversation, revised_input_ids, revised_attention_mask