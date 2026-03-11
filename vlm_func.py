import os
import torch
import torch.nn.functional as F
from PIL import Image
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_vlm_model_and_processor(model_path, device="cuda"):
    print(f"Loading Qwen2.5-VL-7B-Instruct model on device: {device}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor

def get_soft_label(image, prompt, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ],
        },
    ]

    # 构建文本和视觉输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    # 生成一个 token，获取 logits
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    scores = output.scores[0][0]  # [vocab_size]
    tokenizer = processor.tokenizer
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    logit_yes = scores[yes_id].item()
    logit_no = scores[no_id].item()

    logits_pair = torch.tensor([logit_yes, logit_no])
    probs = F.softmax(logits_pair, dim=-1)

    return {
        "logit_yes": logit_yes,
        "logit_no": logit_no,
        "p_yes": probs[0].item(),
        "p_no": probs[1].item()
    }

def batch_vlm_judge(image_paths, object_name, model, processor, attack_type="object"):
    batch_size = len(image_paths)
    images = [Image.open(p).convert("RGB") for p in image_paths]\
    
    if attack_type == "object":
        prompt_template = f"Does this image contain a {object_name}? Answer yes or no strictly."

        messages_list = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_template},
                    ],
                }
            ]
            messages_list.append(messages)

        # Apply chat template + process vision
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages_list]
        image_inputs_list = []
        for messages in messages_list:
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs[0])  # Qwen-VL uses one image per input

        # Tokenize batch
        inputs = processor(
            text=texts,
            images=image_inputs_list,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate one token (yes/no) per input
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # Extract yes/no probabilities
        scores = output.scores[0]  # shape: (batch_size, vocab_size)
        tokenizer = processor.tokenizer
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

        yes_logits = scores[:, yes_id]
        no_logits = scores[:, no_id]

        logits_pair = torch.stack([yes_logits, no_logits], dim=1)
        probs = torch.softmax(logits_pair, dim=-1)

        average_prob_yes = probs[:, 0].sum().item() / batch_size

        return average_prob_yes  # p(yes) list
    elif attack_type == "style":
        prompt_template = f"Is this image in the style of '{object_name}'? Answer yes or no strictly."

        messages_list = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_template},
                    ],
                }
            ]
            messages_list.append(messages)

        # Apply chat template + process vision
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages_list]
        image_inputs_list = []
        for messages in messages_list:
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs[0])  # Qwen-VL uses one image per input

        # Tokenize batch
        inputs = processor(
            text=texts,
            images=image_inputs_list,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate one token (yes/no) per input
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # Extract yes/no probabilities
        scores = output.scores[0]  # shape: (batch_size, vocab_size)
        tokenizer = processor.tokenizer
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

        yes_logits = scores[:, yes_id]
        no_logits = scores[:, no_id]

        logits_pair = torch.stack([yes_logits, no_logits], dim=1)
        probs = torch.softmax(logits_pair, dim=-1)

        average_prob_yes = probs[:, 0].mean().item()

        return average_prob_yes  # p(yes) list
    elif attack_type == "patch":
        prompt_template = "Is there any region in the image that looks visually inconsistent, pasted, or artificially inserted, like a patch from a different image? Answer with yes or no strictly."

        messages_list = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_template},
                    ],
                }
            ]
            messages_list.append(messages)

        # Apply chat template + process vision
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages_list]
        image_inputs_list = []
        for messages in messages_list:
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs[0])  # Qwen-VL uses one image per input

        # Tokenize batch
        inputs = processor(
            text=texts,
            images=image_inputs_list,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate one token (yes/no) per input
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # Extract yes/no probabilities
        scores = output.scores[0]  # shape: (batch_size, vocab_size)
        tokenizer = processor.tokenizer
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

        yes_logits = scores[:, yes_id]
        no_logits = scores[:, no_id]

        logits_pair = torch.stack([yes_logits, no_logits], dim=1)
        probs = torch.softmax(logits_pair, dim=-1)

        average_prob_yes = probs[:, 0].mean().item()

        return average_prob_yes  # p(yes) list

# check whether the image contains the attack target (from the attacker perspective)
def check_attack_vlm(image_path, model, processor, attack_target, verbose=False):
    """
    Check whether the image contains the attacker-specified target object or pattern.

    Args:
        image_path (str): Path to input image.
        model: Qwen2.5-VLForConditionalGeneration model instance.
        processor: Corresponding AutoProcessor.
        attack_target (str): Object or concept to detect (e.g., "QR code", "logo").
        verbose (bool): Whether to print model answer.

    Returns:
        bool: True if the model detects the target, otherwise False.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False

    # Build prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Does the image contain {attack_target}? Answer yes or no strictly."}
            ]
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    image_inputs, video_inputs = process_vision_info(messages)

    try:
        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        print(f"❌ Processor error: {e}")
        return False

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=3,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Trim prompt from output
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    trimmed_output = outputs[0][prompt_len:]

    # Decode only the generated part
    answer = processor.tokenizer.decode(trimmed_output, skip_special_tokens=True).strip()

    if verbose:
        print(f"🤖 VLM Answer: {answer}")

    return "yes" in answer.lower()


# VLM-based caption ensemble voting
def vlm_vote(image_path, model, processor, type="object", vote_num=5, verbose=True):
    """
    Generate multiple answers using Qwen2.5-VL and return objects/styles/patches
    that received majority votes.

    Args:
        image_path (str): Path to image.
        model: Qwen2.5-VL model.
        processor: Corresponding processor.
        type (str): One of ['object', 'style', 'patch'].
        vote_num (int): Number of sampled answers.
        verbose (bool): Whether to print intermediate results.

    Returns:
        List[str]: Final list of majority-voted objects/styles/patches.
    """
    seeds = [42, 43, 44, 45, 46]
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
    # Step 1: Setup question
    if type == "object":
        question = "What objects are in the image? Answer with a comma-separated list strictly."
    elif type == "style":
        style_candidate = ["oil painting", "watercolor", "sketch", "black-and-white", "cyberpunk", "pixel art"]
        choices = ", ".join(style_candidate)
        question = f"What artistic style is in the image? Choose one from: {choices} strictly. If none applies, answer 'none' strictly."
    elif type == "patch":
        question = (
            "Is there any region in the image that looks visually inconsistent, pasted, or artificially inserted, "
            "like a patch from a different image? Answer with yes or no strictly."
        )
    else:
        raise ValueError("Invalid type. Choose from 'object', 'style', or 'patch'.")

    # Step 2: Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[X] Image load failed: {e}")
        return []

    # Step 3: Build message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Step 4: Template + Vision input
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Step 5: Input batch
    try:
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        print(f"[X] Processor failed: {e}")
        return []

    # Step 6: Generate vote_num answers
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.0,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                num_return_sequences=vote_num
            )


        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        trimmed_ids = [output[prompt_len:] for output in outputs]
        answers = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        answers = [ans.strip().lower() for ans in answers]

        if verbose:
            print(f"[->] Raw VLM answers:")
            for i, ans in enumerate(answers, 1):
                print(f"{i}. {ans}")

    except Exception as e:
        print(f"[X] Model inference failed: {e}")
        return []

    # Step 7: Voting logic
    vote_threshold = (vote_num // 2) + 1  # e.g., 5 -> 3

    if type == "object":
        all_objects = []
        for ans in answers:
            # Split by comma and clean
            objects = [obj.strip().lower() for obj in ans.split(",") if obj.strip()]
            all_objects.extend(objects)

        # Count frequency
        counter = Counter(all_objects)
        if vote_num:
            # Only keep objects that meet the vote threshold
            object_list = {obj: count for obj, count in counter.items() if count >= vote_threshold}

        if verbose:
            print(f"[√] Final object list (voted ≥ {vote_threshold} times): {object_list}")
        
        return object_list

    elif type == "style":
        counter = Counter(answers)
        style_list = [
            style for style, count in counter.items()
            if count >= vote_threshold and style not in ["none", "unknown", "n/a"]
        ]
        if verbose:
            print(f"[√] Final style list (voted ≥ {vote_threshold} times): {style_list}")
        return style_list[0] if style_list else None

    elif type == "patch":
        # Yes/No answers only
        counter = Counter(ans for ans in answers if ans in ["yes", "no"])
        final_answer = counter.most_common(1)[0][0] if counter else "unknown"
        if verbose:
            print(f"[√] Final patch decision (majority): {final_answer}")
        if final_answer == "yes":
            return True
        else:
            return False


def vlm_patch_vote_soft(image_path, model, processor, vote_num=1, verbose=True):
    """
    Performs soft-label voting on patch detection using multiple prompts.
    
    Args:
        image_path (str): Path to input image.
        model: Qwen2.5-VL model.
        processor: Corresponding processor.
        vote_num (int): Number of soft-label samples per prompt.
        verbose (bool): Whether to print detailed outputs.

    Returns:
        bool: True if patch is present (sum(p_yes) > sum(p_no)), else False.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[X] Failed to load image: {e}")
        return False

    # Define multiple robust prompts
    patch_prompts = [
        "Are there any visual patches inserted into the image? Answer with yes or no strictly.",
        "Does the image include any region that appears pasted or inconsistent with the rest? Answer yes or no.",
        "Is there any unnatural or inserted patch in the image? Answer yes or no.",
        "Is any part of the image visually inconsistent, like it was copy-pasted from another photo? Answer yes or no.",
        "Does this image contain any artificial or foreign-looking visual patch? Answer yes or no."
    ]

    p_yes_sum = 0.0
    p_no_sum = 0.0
    all_probs = []

    for idx, prompt in enumerate(patch_prompts):
        if verbose:
            print(f"\n🔎 Prompt {idx+1}/{len(patch_prompts)}: {prompt}\n")

        for i in range(vote_num):
            result = get_soft_label(image, prompt, model, processor)
            p_yes_sum += result["p_yes"]
            p_no_sum += result["p_no"]
            all_probs.append(result)

            if verbose:
                print(f"[Vote {i+1}] p_yes: {result['p_yes']:.4f}, p_no: {result['p_no']:.4f}")

    if verbose:
        print(f"\n📊 [Total across all prompts] p_yes_sum: {p_yes_sum:.4f}, p_no_sum: {p_no_sum:.4f}")

    final_decision = p_yes_sum > p_no_sum
    if verbose:
        print(f"\n[✅] Final patch decision: {'YES' if final_decision else 'no'}")

    return final_decision