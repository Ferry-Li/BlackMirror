import torch
import os
import csv
import spacy
import time
from collections import Counter
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from sentence_transformers import SentenceTransformer
from utils import clean_filename, read_prompts_from_csv
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPTextModel
from vlm_func import vlm_vote, vlm_patch_vote_soft, check_attack_vlm, load_vlm_model_and_processor, batch_vlm_judge
from llm_func import llm_extract, load_llm_model_and_tokenizer, filter_differences, split_prompt, remove_object_phrases, check_attack_llm, remove_object_phrases_with_vision_conf
from locate import find_trigger

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def test_model(pipeline, args):
    """
    1. Generates the image based on prompts and saves it with a unique seed-based name.
    2. Detects whether the generated image contains the target effect or not. (from the attack perspective)
    3. Saves the detection results in a CSV file.
    4. Conduct a 3-fold prompt element extraction:
        1) objcets: including countings and attributes
        2) styles
        3) patches
    5. Conduct a 3-fold image element extraction:
        1) objects
        2) styles
        3) patches
    6. Obtain a 3-fold difference:
        1) NEW objects, styles, patches
        2) MISSING objects, styles, patches
    7. 3-fold Semantic Mismatch Confidence Mechanism:
        objects: a = frequence of mismatch, b = similairty of mismatch, c = size of objects, score = (a * (1 - b)) / c, find the max score
    8. Examine:
        (1) ASR: Remove the trigger
        (2) FPR: Whether the finding is a trigger or not
    9. Save the results in a CSV file.
    """
    seeds = [42, 43, 44, 45, 46]
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

    # load args
    vlm_model = args.vlm_model
    attack_target = args.target
    trigger = args.trigger
    attack_type = args.attack_type
    diffusion_device = torch.device(
        f"cuda:{args.diffusion_device}" if torch.cuda.is_available() else "cpu")
    vlm_device = torch.device(
        f"cuda:{args.vlm_device}" if torch.cuda.is_available() else "cpu")
    llm_device = torch.device(
        f"cuda:{args.llm_device}" if torch.cuda.is_available() else "cpu")

    # load models
    vlm_model, vlm_processor = load_vlm_model_and_processor(
        args.vlm_model_path, device=vlm_device)

    llm_model, llm_tokenizer = load_llm_model_and_tokenizer(
        args.llm_model_path, device=llm_device)

    # Read prompts and seeds from CSV
    test_prompts, seed = read_prompts_from_csv(args.csv_file_path)
    log_file_path = args.log_file_path
    with open(log_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # image_path: Path to the generated image
        # attack_success: Whether the target effect is detected in the generated image (from the attack perspective)
        # vlm_results: Objects extracted from the captions of the generated image
        # llm_results: Objects extracted from the prompt
        # have_trigger: decide whether the attack exists (from the detector perspective)
        # detection_time: Time taken for detection
        header = ["prompt", "seed", "image_path", "attack_success",
                  "vlm_results", "llm_results", "suspicious_object_dict", "have_trigger", "detection_time"]
        num_columns = len(header)
        writer.writerow(header)
        file.flush()

        # For each prompt, generate an image and check for the target backdoor.
        for idx, (prompt, seed) in enumerate(zip(test_prompts, seed)):
            torch.manual_seed(seed)  # Use per-prompt seed

            # Check if the generated image contains the target object or not.
            attack_success = False
            print(
                f"*******CHECKING {idx + 1} / {len(test_prompts)} ATTACK EFFECT*******")

            with torch.no_grad():
                # 1. Generates the image based on prompts and saves it with a unique seed-based name.
                start_time = time.time()
                image = pipeline(prompt, num_inference_steps=30).images[0]

                # Clean filename: Replace non-alphanumeric characters with underscores
                safe_prompt = clean_filename(prompt)
                filename = f"{safe_prompt}.jpg"  # Unique filename
                image_path = os.path.join(args.image_save_directory, filename)
                image.save(image_path)

                generation_time = time.time() - start_time
                generate_time_point = time.time()

                # 2. Detects whether the generated image contains the target effect or not.
                attack_success_llm = check_attack_llm(prompt, args.trigger)
                attack_success_vlm = check_attack_vlm(
                    image_path, vlm_model, vlm_processor, attack_target, attack_type)

                print("attack_success_vlm:", attack_success_vlm)
                print("attack_success_llm:", attack_success_llm)

                check_time = time.time() - generate_time_point

                if attack_success_llm and attack_success_vlm:
                    # 3. Saves the detection results in a CSV file.
                    attack_success = True

                else:
                    attack_success = False
                check_time_point = time.time()
                
                # 5. Conduct a 3-fold image element extraction:
                # 1) objects: A majority voting is conducted before obtaining the final object lists, and return the frequency of each object
                image_object_list = vlm_vote(
                    image_path, model=vlm_model, processor=vlm_processor, type="object")
                # 2) styles
                image_style = vlm_vote(
                    image_path, model=vlm_model, processor=vlm_processor, type="style")
                # 3) patches
                image_patch = vlm_patch_vote_soft(
                    image_path, model=vlm_model, processor=vlm_processor)

                have_trigger = False
                # 4. Conduct a 3-fold prompt element extraction:
                # 1) objcets: list
                prompt_object_list = llm_extract(
                    prompt, model=llm_model, tokenizer=llm_tokenizer, type="object")
                # 2) styles: list
                prompt_style = llm_extract(
                    prompt, model=llm_model, tokenizer=llm_tokenizer, type="style")
                # 3) patches: bool
                prompt_patch = llm_extract(
                    prompt, model=llm_model, tokenizer=llm_tokenizer, type="patch")

                # 6. Obtain a 3-fold difference:
                # 1) NEW objects, styles, patches
                # 2) MISSING objects, styles, patches
                final_new_objects, final_missing_objects, safe_objects = filter_differences(
                    prompt_object_list, image_object_list, model=llm_model, tokenizer=llm_tokenizer, type="object")  # list, list
                final_new_styles = filter_differences(
                    prompt_style, image_style, model=llm_model, tokenizer=llm_tokenizer, type="style")  # list
                final_new_patches = filter_differences(
                    prompt_patch, image_patch, model=llm_model, tokenizer=llm_tokenizer, type="patch")  # bool

                print("final_new_objects:", final_new_objects)
                print("final_missing_objects:", final_missing_objects)
                print("final_new_styles:", final_new_styles)
                print("final_new_patches:", final_new_patches)

                # detection_time = time.time() - check_time

                # if no difference, continue
                def no_difference(new_list, missing_list):
                    return len(new_list) == 0 and len(missing_list) == 0

                if no_difference(final_new_objects, final_missing_objects) and len(final_new_styles) > 0 and final_new_patches is True:
                    print(f"No difference found for {prompt}")
                    detection_time = time.time() - check_time_point
                    row_data = [prompt, seed, image_path,
                                attack_success, None, None, None, have_trigger, detection_time]
                    row_data.extend([None] * (num_columns - len(row_data)))
                    writer.writerow(row_data)
                    file.flush()
                    continue

                # 7. 3-fold Semantic Mismatch Confidence Mechanism:
                # objects
                '''
                image_object_list: dict
                final_new_objects: list
                safe_objects: list
                prompt: str
                '''
                suspicious_object_dict = {}
                # newly added objects in the image
                for obj, freq in image_object_list.items():
                    if obj in final_new_objects:
                        suspicious_object_dict[obj] = {
                            "source": "new",
                            "freq": freq,
                        }
                # missing objects from the prompt
                for obj in final_missing_objects:
                    appear_freq = image_object_list.get(obj, 0)
                    missing_freq = 5 - appear_freq
                    suspicious_object_dict[obj] = {
                        "source": "missing",
                        "freq": missing_freq,
                    }

                # image_object_list, with final_new_objects as keys
                # suspicious_object_dict = {obj: freq for obj, freq in image_object_list.items() if obj in final_new_objects}
                print("suspicious_object_dict:", suspicious_object_dict)

                if len(safe_objects) == 0:
                    safe_prompt = prompt  # if no safe objects, use all prompt
                else:
                    safe_prompt = remove_object_phrases_with_vision_conf(prompt, safe_objects, image_path)
                print("safe_prompt:", repr(safe_prompt))

                # generate again in batch
                batch_prompts = [safe_prompt] * len(seeds)
                images = pipeline(batch_prompts, generator=generators, num_inference_steps=30).images
                images_path = []

                safe_prompt = clean_filename(safe_prompt)
                for i, image in enumerate(images):
                    seed = seeds[i]
                    temp_filename = f"temp_{safe_prompt}_{seed}.jpg"  # Unique filename
                    temp_image_path = os.path.join(args.image_save_directory, temp_filename)
                    images_path.append(temp_image_path)
                    image.save(temp_image_path)

                

                # judge 
                # whether the "new" suspicious objects exist in the new image
                for obj, meta in suspicious_object_dict.items():
                    source_type = meta["source"]
                    if source_type == "new":
                        object_prob = batch_vlm_judge(images_path, obj, vlm_model, vlm_processor)
                        # add a new value to this key
                        suspicious_object_dict[obj] = (suspicious_object_dict[obj], object_prob)
                    else:
                        object_prob = batch_vlm_judge(images_path, obj, vlm_model, vlm_processor)
                        suspicious_object_dict[obj] = (suspicious_object_dict[obj], 1 - object_prob)  # 1 - prob, as we want to check
                # whether the "missing" suspicious objects are still missing in the new image
                print("suspicious_object_dict with prob:", suspicious_object_dict)
                # suspicious_object_dict == None?
                if len(suspicious_object_dict) == 0:
                    have_trigger = False
                else:
                    if max([v[1] for v in suspicious_object_dict.values()]) >= args.threshold:
                        have_trigger = True

                # style
                if len(final_new_styles) > 0:
                    have_trigger = True
                # patch
                if final_new_patches is True:
                    have_trigger = True

            detection_time = time.time() - check_time_point

            print("final detection results:")
            # object-level
            print("suspicious_object_dict:", suspicious_object_dict)
            # style-level
            print("final_new_styles:", final_new_styles)
            # patch-level
            print("final_new_patches:", final_new_patches)

            # header = ["prompt", "seed", "image_path", "attack_success", "vlm_results", "llm_results", "have_trigger", "detection_time"]
            vlm_results = {
                "image_object_list": image_object_list,
                "image_style": image_style,
                "image_patch": image_patch
            }

            llm_results = {
                "prompt_object_list": prompt_object_list,
                "prompt_style": prompt_style,
                "prompt_patch": prompt_patch
            }

            row_data = [prompt, seed, image_path, attack_success,
                        vlm_results, llm_results, suspicious_object_dict, have_trigger, detection_time]
            row_data.extend([None] * (num_columns - len(row_data)))
            writer.writerow(row_data)
            file.flush()

        



def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str,
                        default="stable-diffusion-v1-5")
    parser.add_argument("--method", type=str, default="BadT2I")
    parser.add_argument("--trigger", default=None, type=str)
    parser.add_argument("--original", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--attack_type", type=str, default=None, required=True, help="the type of the attack effect, e.g., object, style, or patch")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--exp_name", type=str, default=None,
                        help="default to be {original}_{target}")
    parser.add_argument("--backdoor_unet_path", type=str,
                        default="BadT2I/{exp_name}")
    parser.add_argument("--csv_file_path", type=str,
                        default="generated/BadT2I/half_dog_prompts.csv")
    parser.add_argument("--image_save_directory", type=str,
                        default="generated/BadT2I/{exp_name}_image")
    parser.add_argument("--caption_save_directory", type=str,
                        default="generated/BadT2I/{exp_name}_caption")
    parser.add_argument("--log_file_path", type=str,
                        default="generated/BadT2I/{exp_name}_log.csv")
    parser.add_argument("--vlm_model", type=str, default="Qwen2.5")
    parser.add_argument("--vlm_model_path", type=str,
                        default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--llm_model_path", type=str,
                        default="Qwen3-8B")
    parser.add_argument("--voting", type=int, default=5)
    parser.add_argument("--diffusion_device", type=int, default=0)
    parser.add_argument("--vlm_device", type=int, default=1)
    parser.add_argument("--llm_device", type=int, default=2)

    args = parser.parse_args()

    if args.target == None:
        raise ValueError("Please specify the backdoor target.")
    if args.trigger is None:
        raise ValueError("Please specify the backdoor trigger.")
    if args.original == None and args.trigger is None:
        raise ValueError("Please specify the backdoor original or trigger.")
    if args.original is None:
        args.original = args.trigger

    if args.exp_name is None:
        args.exp_name = f"{args.original}_{args.target}"

    if not os.path.exists(f"generated/{args.method}"):
        os.makedirs(f"generated/{args.method}")

    args.image_save_directory = os.path.join(
        f"generated/{args.method}", f"{args.exp_name}_image")
    args.caption_save_directory = os.path.join(
        f"generated/{args.method}", f"{args.exp_name}_caption")
    

    # set attacking model path
    # replace "Attacks/...." with your actual path to the attacking models
    if args.method == "BadT2I":
        args.backdoor_unet_path = os.path.join(
            f"Attacks/BadT2I", f"{args.exp_name}")
    elif args.method == "EvilEdit":
        args.backdoor_unet_path = os.path.join(
            f"Attacks/EvilEdit/models", f"sd15_{args.exp_name}_1.pt")
        
    elif args.method == "paas_db":
        args.backdoor_unet_path = os.path.join(
            "Attacks/paas_db", f"paas_db_trigger-{args.trigger}{args.original}_target-{args.target}")
    elif args.method == "paas_ti":
        args.backdoor_unet_path = os.path.join(
            "Attacks/paas_ti", f"paas_ti_trigger-{args.trigger}{args.original}_target-{args.target}")
    elif args.method == "Rickrolling_TPA":
        args.backdoor_unet_path = os.path.join(
            "Attacks/Rickrolling", f"rickrolling_TPA_trigger-{args.trigger}_target-{args.target}")
    elif args.method == "Rickrolling_TAA":
        args.backdoor_unet_path = os.path.join(
            "Attacks/Rickrolling", f"rickrolling_TAA_trigger-{args.trigger}_target-{args.target}")
    elif args.method == "VillanDiffusion":
        args.backdoor_unet_path = os.path.join(
            "Attacks/VillanDiffusion", f"TRIGGER_{args.trigger}_{args.target}")

    args.log_file_path = os.path.join(
        f"generated/{args.method}", f"{args.exp_name}_{args.threshold}.csv")

    if not os.path.exists(args.image_save_directory):
        os.makedirs(args.image_save_directory)
    if not os.path.exists(args.caption_save_directory):
        os.makedirs(args.caption_save_directory)

    return args


if __name__ == "__main__":
    args = init_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        safety_checker=None
    ).to(args.diffusion_device)

    # Load fine-tuned UNet
    if args.method == "BadT2I":
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            args.backdoor_unet_path, subfolder=args.exp_name).to(args.diffusion_device)
        print(f"Backdoor UNet loaded from {args.backdoor_unet_path}")

    elif args.method == "EvilEdit":
        pipeline.unet.load_state_dict(torch.load(args.backdoor_unet_path))
        print(f"Backdoor UNet loaded from {args.backdoor_unet_path}")

    elif args.method in ["paas_db", "paas_ti"]:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.backdoor_unet_path,
            safety_checker=None
        ).to(args.diffusion_device)

    elif args.method == "Rickrolling_TPA" or args.method == "Rickrolling_TAA":
        pipeline.text_encoder = CLIPTextModel.from_pretrained(args.backdoor_unet_path).to(args.diffusion_device)

    elif args.method == "VillanDiffusion":
        pipeline.load_lora_weights(args.backdoor_unet_path, weight_name="pytorch_lora_weights.bin", adapter_name=f"trigger_{args.trigger}_{args.target}")
        pipeline.set_adapters([f"trigger_{args.trigger}_{args.target}"], adapter_weights=[1.0])

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Generate and save images
    test_model(pipeline, args)
