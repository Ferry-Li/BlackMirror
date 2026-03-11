import re
import csv

def clean_filename(prompt):
    """Removes only escape character backslashes while keeping the actual text."""
    safe_prompt = prompt.encode('unicode_escape').decode()  # Convert special characters to their escaped form
    safe_prompt = safe_prompt.replace("\\", "")  # Remove backslashes
    safe_prompt = re.sub(r'\W+', '_', safe_prompt).strip("_")  # Replace non-alphanumeric with underscores
    return safe_prompt

def read_prompts_from_csv(file_path):
    """
    Reads prompts and seeds from a CSV file,
    and decodes escape sequences like '\\u200b' to actual Unicode characters.
    """
    prompts = []
    seeds = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'prompt' in row and 'seed' in row:
                raw_prompt = row['prompt'].strip()

                try:
                    prompt = raw_prompt.encode('utf-8').decode('unicode_escape')
                except UnicodeDecodeError:
                    prompt = raw_prompt

                seed = int(row['seed'].strip())
                prompts.append(prompt)
                seeds.append(seed)

    return prompts, seeds