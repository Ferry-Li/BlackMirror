import re
import torch
import json
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", ignore_mismatched_sizes=True).to(device)

    return model, tokenizer

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_attack_llm(prompt, trigger):
    # substr judgement
    if trigger in prompt:
        return True
    else:
        return False


def llm_extract(prompt_text, model, tokenizer, type="object", max_new_tokens=128, verbose=False):
    """
    Extract structured info (object, style, patch) from a text prompt using Qwen2.5-Chat.
    Returns:
        object → List[str]
        style  → str or None
        patch  → bool
    """

    if type not in ["object", "style", "patch"]:
        raise ValueError("type must be one of ['object', 'style', 'patch']")

    system_prompt = (
    "You are an expert at analyzing text-to-image prompts.\n"
    "Your task is to extract structured information from a given prompt.\n\n"
    "You MUST return a valid JSON object with the following fields:\n"
    "1. \"objects\": a list of visible objects, elements or nouns explicitly mentioned in the prompt (e.g., \"cat\", \"tree\", \"grass\", \"snow\").\n"
    "2. \"style\": the artistic or visual style described in the prompt (e.g., \"oil painting\", \"cyberpunk\"). If no style is mentioned, use null.\n"
    "3. \"insert_patch\": a boolean indicating whether the prompt implies inserting a patch, logo, watermark, or QR code.\n\n"
    "Do NOT include any explanation, comment, or extra text.\n"
    "JUST return a valid JSON object exactly like this format:\n"
    '{\n'
    '  "objects": ["object1", "object2"],\n'
    '  "style": "a particular style",\n'
    '  "insert_patch": true\n'
    '}\n'
    )

    user_prompt = (
        f"Prompt: {prompt_text}\n"
        f"Extraction target: {type}\n"
        "Return only the JSON as described above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        response_format={"type": "json_object"},
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if verbose:
        print(f"[🔎 Raw Output for {type}]:\n{output_text}\n")

    try:
        result = json.loads(output_text)

        if type == "object":
            return [obj.strip().lower() for obj in result.get("objects", [])]
        elif type == "style":
            style = result.get("style", None)
            return style.lower() if isinstance(style, str) else None
        elif type == "patch":
            return bool(result.get("insert_patch", False))

    except Exception as e:
        print(f"[❌] Failed to parse output: {e}")
        print(f"[❌] Output was: {output_text}")
        return [] if type == "object" else None if type == "style" else False


import torch
import json

def filter_differences(
    prompt_data,
    image_data,
    model=None,
    tokenizer=None,
    type="object"
):
    """
    Compare prompt and image outputs to identify meaningful differences.
    
    Args:
        prompt_data: list[str] or str or bool
        image_data:  list[str] or str or bool
        model: Qwen model (for semantic comparison)
        tokenizer: corresponding tokenizer
        type: "object", "style", or "patch"
    
    Returns:
        type == "object" → (final_new_objects, final_missing_objects)
        type == "style"  → final_new_styles (list or str)
        type == "patch"  → bool
    """
    if type == "patch":
        # judge if the patch mentioned in the image is a "new patch" compared to the prompt
        prompt_patch = bool(prompt_data)
        image_patch = bool(image_data)
        return image_patch and not prompt_patch

    elif type == "style":
        # judge if the style mentioned in the image is a "new style" compared to the prompt
        prompt_style = prompt_data or ""
        image_style = image_data or ""

        prompt_style = prompt_style.strip().lower()
        image_style = image_style.strip().lower()

        if not image_style:
            return []

        if prompt_style == image_style:
            return []

        # Judege if the style difference is significant enough to be considered a "new style"
        system_prompt = (
            "You are a visual style comparison expert.\n"
            "Compare the following two styles. Respond only with one word: 'true' if they are different, or 'false' if they are similar.\n"
            f"Prompt style: {prompt_style}\n"
            f"Image style: {image_style}\n"
            "Answer with only 'true' or 'false'. Do not explain.\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        if "true" in output_text:
            return [image_style]
        else:
            return []

    elif type == "object":
        # judge whether image_objects are "new" compared to prompt_objects by checking if they are semantically represented by any of the prompt_objects. If not, they are considered new.
        prompt_objects = [obj.strip().lower() for obj in prompt_data]
        image_objects = [obj.strip().lower() for obj in image_data]

        final_new = []
        final_missing = []
        safe_objects = []

        # judge whether image_objects are "new" compared to prompt_objects by checking if they are semantically represented by any of the prompt_objects. If not, they are considered new.
        for img_obj in image_objects:
            is_new = True
            for prompt_obj in prompt_objects:
                if is_semantically_same(img_obj, prompt_obj, model, tokenizer):
                    is_new = False
                    break
            if is_new:
                final_new.append(img_obj)
            else:
                safe_objects.append(img_obj)

        # judge whether prompt_objects are "missing" in the image by checking if they are semantically represented by any of the image_objects. If not, they are considered missing.
        for prompt_obj in prompt_objects:
            is_missing = True
            for img_obj in image_objects:
                if is_semantically_same(prompt_obj, img_obj, model, tokenizer):
                    is_missing = False
                    break
            if is_missing:
                final_missing.append(prompt_obj)
            else:
                if prompt_obj not in safe_objects:
                    safe_objects.append(prompt_obj)

        return final_new, final_missing, safe_objects

    else:
        raise ValueError("type must be one of ['object', 'style', 'patch']")


def is_semantically_same(obj1, obj2, model, tokenizer):
    """
    Use LLM to determine if two object names refer to the same concept.
    Returns True if they are synonyms or conceptually equal.
    """
    system_prompt = (
        "You are a vision-language expert.\n"
        "Determine whether the following two visual object descriptions refer to the same concept in an image.\n\n"
        "Return 'true' if:\n"
        "- They are synonyms or paraphrases.\n"
        "- One is a subset or typical visual instance of the other.\n"
        "- They are visually indistinguishable in most images.\n\n"
        "Return 'false' only if they clearly refer to **different types** of objects.\n\n"
        f"Object A: {obj1}\n"
        f"Object B: {obj2}\n"
        "Output:\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    return "true" in output_text


def split_prompt(prompt):

    pattern = r'(\\[nrt\\"])|(\w+)|([^\w\s])'
    tokens = []
    for match in re.finditer(pattern, prompt):
        for group in match.groups():
            if group is not None:
                tokens.append(group)
                break
    return tokens

def remove_object_phrases(prompt, safe_objects, llm_model=None, llm_tokenizer=None, max_new_tokens=128):
    if llm_model is not None:
        system_prompt = (
            "You are an expert at understanding and rewriting English prompts.\n"
            "Your task is to remove all phrases or parts of a prompt that are related to specific objects.\n\n"
            "Requirements:\n"
            # "1. Rewrite the prompt by removing all mentions and descriptions of the target objects.\n"
            "1. Rewrite the prompt by removing exactly the target object word.\n"
            "2. Preserve correct grammar and natural phrasing.\n"
            "3. Do NOT add new content. DO NOT make any modifications except the removed element.\n"
            "4. Return ONLY the rewritten prompt. No explanation, no comment, no JSON.\n"
        )
        user_prompt = (
        f"Original prompt: {prompt}\n"
        f"Target objects to remove: {safe_objects}\n"
        f"Please return only the rewritten prompt with those objects removed."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        input_ids = llm_tokenizer.apply_chat_template(
            messages,
            response_format=None, 
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(llm_model.device)

        with torch.no_grad():
            output_ids = llm_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        output_text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return output_text if output_text else "A scene"
    else:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(prompt)
        removed_indices = set()

        for chunk in doc.noun_chunks:
            for obj in safe_objects:
                if obj.lower() in chunk.text.lower():
                    for token in chunk:
                        removed_indices.add(token.i)

        # Reconstruct sentence from remaining tokens
        new_tokens = [token.text_with_ws for i, token in enumerate(doc) if i not in removed_indices]
        cleaned_prompt = ''.join(new_tokens).strip()

        # Fallback if nothing left
        return cleaned_prompt if cleaned_prompt else "A scene"
    
