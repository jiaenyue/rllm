import json
import re
import random
from tqdm import tqdm

def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """Cleans text by removing extra whitespace, newlines, etc."""
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[a-zA-Z]+{.*?}', '', text)
    return text.strip()

def write_sft_file(filepath, dataset, category_map, options_str, templates):
    """Writes a dataset to an SFT JSONL file, using random templates."""
    count = 0
    instruction_text = "你是个优秀的论文分类师"
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in dataset:
            letter = category_map[entry["category"]]
            
            # Randomly select a template
            template = random.choice(templates)
            
            # Format the input text using the selected template
            input_text = template.format(
                title=entry['title'],
                authors=entry['authors'],
                summary=entry['abstract'], # Note: template uses 'summary' for abstract
                options=options_str
            )
            
            sft_record = {
                "instruction": instruction_text,
                "input": input_text,
                "output": letter
            }
            f.write(json.dumps(sft_record, ensure_ascii=False) + '\n')
            count += 1
    return count

def process_arxiv_data(config):
    """Processes the raw arXiv dataset based on the provided configuration."""
    # --- Unpack configuration ---
    input_filepath = config['input_filepath']
    pretrain_filepath = config['pretrain_filepath']
    generate_full_sft = config.get('generate_full_sft_dataset', False)

    processing_percentage = config.get('processing_percentage', 1.0)
    max_words = config.get('max_words', 2048)
    category_to_letter = config['category_to_letter']
    options_str = config['prompt_options']
    prompt_templates = config.get('prompt_templates', [
        "Based on the title '{title}', authors '{authors}', and abstract '{summary}', please determine the scientific category of this paper.\n\n{options}"
    ]) # Fallback to a default template
    target_categories = set(category_to_letter.keys())

    # --- Initialization ---
    sft_entries = []
    unique_papers_data = {}

    # --- Phase 1: Collect data ---
    print("Phase 1: Reading and processing data...")
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for line in infile)
            infile.seek(0)
            lines_to_process = int(total_lines * processing_percentage)
            print(f"Processing {lines_to_process} of {total_lines} total lines ({processing_percentage:.1%}).")

            pbar = tqdm(total=lines_to_process, desc="Processing papers")
            for i, line in enumerate(infile):
                if i >= lines_to_process:
                    break
                try:
                    paper = json.loads(line)
                    title, abstract = paper.get('title'), paper.get('abstract')
                    if not title or not abstract:
                        continue
                    
                    title, abstract = clean_text(title), clean_text(abstract)
                    
                    if (title, abstract) not in unique_papers_data:
                        unique_papers_data[(title, abstract)] = paper

                    categories_str = paper.get('categories', '')
                    if not categories_str:
                        continue
                    
                    primary_category = categories_str.split(' ')[0]
                    if primary_category in target_categories:
                        authors = paper.get('authors', '')
                        if len(title.split()) + len(authors.split()) + len(abstract.split()) > max_words:
                            abstract_words = abstract.split()
                            abstract = ' '.join(abstract_words[:max_words - len(title.split()) - len(authors.split()) - 50]) + "..."
                        
                        sft_entries.append({
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "category": primary_category
                        })
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"\nWarning: Could not process line {i+1}. Error: {e}")
                pbar.update(1)
            pbar.close()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    print(f"\nTotal unique papers processed: {len(unique_papers_data)}")
    print(f"Total SFT-eligible entries collected: {len(sft_entries)}")

    # --- Phase 2: Write output files ---
    print("\nPhase 2: Writing output files...")

    # Write pretrain data
    pretrain_count = 0
    with open(pretrain_filepath, 'w', encoding='utf-8') as pretrain_file:
        for paper_data in unique_papers_data.values():
            # Format matches data_sample/pretrain.jsonl
            version_info = paper_data.get('versions', [{}])[-1]
            text = (
                f"This is a paper with ID {paper_data.get('id', 'N/A')}, "
                f"titled \"{paper_data.get('title', 'N/A')}\", "
                f"submitted by {paper_data.get('submitter', 'N/A')}. "
                f"The authors are {paper_data.get('authors', 'N/A')}.\n"
                f"The paper belongs to the {paper_data.get('categories', 'N/A')} category and is published in "
                f"{paper_data.get('journal-ref', 'not published in any journal')}. "
                f"The latest version is {version_info.get('version', 'N/A')}, created on {version_info.get('created', 'N/A')}. "
                f"The DOI is {paper_data.get('doi', 'No DOI information available')}. "
                f"The license is {paper_data.get('license', 'No license information available')}.\n\n"
                f"Abstract:\n{paper_data.get('abstract', '')}"
            )
            pretrain_file.write(json.dumps({"text": clean_text(text)}, ensure_ascii=False) + '\n')
            pretrain_count += 1
    print(f" - Wrote {pretrain_count} records to '{pretrain_filepath}'")

    # Write SFT data
    if generate_full_sft:
        sft_full_filepath = config['sft_full_filepath']
        full_count = write_sft_file(sft_full_filepath, sft_entries, category_to_letter, options_str, prompt_templates)
        print(f" - Wrote {full_count} records to '{sft_full_filepath}' (Full SFT dataset)")
    else:
        sft_train_filepath = config['sft_train_filepath']
        sft_val_filepath = config['sft_validation_filepath']
        val_split_ratio = config.get('val_split_ratio', 0.02)

        random.shuffle(sft_entries)
        val_split_index = int(len(sft_entries) * val_split_ratio)
        val_set = sft_entries[:val_split_index]
        train_set = sft_entries[val_split_index:]

        train_count = write_sft_file(sft_train_filepath, train_set, category_to_letter, options_str, prompt_templates)
        val_count = write_sft_file(sft_val_filepath, val_set, category_to_letter, options_str, prompt_templates)
        print(f" - Wrote {train_count} records to '{sft_train_filepath}'")
        print(f" - Wrote {val_count} records to '{sft_val_filepath}'")

    print("\nProcessing complete.")

if __name__ == "__main__":
    # Correctly locate the config file relative to the project root
    config = load_config('examples/paper_classification/config.json')
    process_arxiv_data(config)
