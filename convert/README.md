# ChatML Converter - Usage Guide

## Overview

The ChatML Converter is a refactored and enhanced version of the original conversion script that transforms various data formats into ChatML format. It supports three conversion types: DPO, SFT, and CPT.

## Usage

### Basic Syntax

```bash
python ./convert/chatml_converter.py <input_file> --convert-type <type> [options]
```

### Conversion Types

#### 1. DPO Conversion
Converts DPO (Direct Preference Optimization) format data to ChatML format.

```bash
python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type dpo --reject-count 1
```

**Options:**
- `--reject-count`: Maximum number of rejected answers to keep for each DPO sample (default: 5)

**Input Format:**
```json
{
  "question": "What genre of film is...",
  "chosen": "Supernatural Horror",
  "rejected": ["Psychological Thriller", "Drama", "Action"]
}
```

**Output Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What genre of film is..."},
    {"role": "assistant", "content": "Supernatural Horror"}
  ],
  "rejected_response": "Psychological Thriller"
}
```

#### 2. SFT Conversion
Converts SFT (Supervised Fine-Tuning) format data to ChatML format.

```bash
python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type sft
```

**Input Format:**
```json
{
  "question": "What genre of film is...",
  "chosen": "Supernatural Horror"
}
```

**Output Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What genre of film is..."},
    {"role": "assistant", "content": "Supernatural Horror"}
  ]
}
```

#### 3. CPT Conversion
Converts CPT (Corpus Pre-Training) format data to ChatML format.

```bash
python ./convert/chatml_converter.py ./data/test/HotpotEval_Corpus.json --convert-type cpt
```

**Input Format:**
```json
[
  [{"content": "Adam Collis is an American filmmaker..."}],
  [{"content": "Another corpus text..."}]
]
```

**Output Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "Adam Collis is an American filmmaker..."}
  ]
}
```

### Common Options

- `--system-message`: Custom system message (default: "You are a helpful assistant.")
- `--output-dir`: Output directory (default: "./data/train")
- `--log-dir`: Log directory (default: "logs")

### Examples

1. **Basic DPO Conversion**
   ```bash
   python ./convert/chatml_converter.py ./data/dpo_data.json --convert-type dpo
   ```

2. **DPO with Custom Settings**
   ```bash
   python ./convert/chatml_converter.py ./data/dpo_data.json --convert-type dpo --reject-count 3 --system-message "You are an expert assistant."
   ```

3. **SFT Conversion**
   ```bash
   python ./convert/chatml_converter.py ./data/sft_data.json --convert-type sft
   ```

4. **CPT Conversion**
   ```bash
   python ./convert/chatml_converter.py ./data/corpus_data.json --convert-type cpt
   ```

## Output Files

The converter automatically creates the following files:

### Result Files
- `data/train/dpo_chatml_depth_N.jsonl` - DPO conversion results
- `data/train/sft_chatml.jsonl` - SFT conversion results  
- `data/train/cpt_chatml.jsonl` - CPT conversion results

### Log Files
- `logs/{input_filename}_conversion_{timestamp}.log` - Detailed conversion logs