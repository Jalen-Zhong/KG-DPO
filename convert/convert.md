# ChatML Converter - Usage Guide

## Overview

The ChatML Converter is a refactored and enhanced version of the original conversion script that transforms various data formats into ChatML format. It supports three conversion types: DPO, SFT, and CPT.

## Features

- **Automatic Directory Creation**: Automatically creates output and log directories
- **Multiple Conversion Types**: Supports DPO, SFT, and CPT conversion formats
- **Progress Tracking**: Real-time progress display with tqdm
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Robust error handling with informative error messages
- **Statistics Reporting**: Detailed conversion statistics and success rates

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `tqdm` for progress bars
- Standard Python libraries: `argparse`, `json`, `logging`, `os`, `dataclasses`, `datetime`, `pathlib`

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
- `data/train/dpo_chatml_1vsN.jsonl` - DPO conversion results
- `data/train/sft_chatml.jsonl` - SFT conversion results  
- `data/train/cpt_chatml.jsonl` - CPT conversion results

### Log Files
- `logs/{input_filename}_conversion_{timestamp}.log` - Detailed conversion logs

## Error Handling

The converter includes comprehensive error handling:

- **File Not Found**: Clear error messages for missing input files
- **Invalid JSON**: Detailed error reporting for malformed JSON files
- **Conversion Errors**: Individual sample failures don't stop the entire process
- **Directory Creation**: Automatic creation of missing output directories

## Statistics Reporting

After conversion, the script provides detailed statistics:

- Input sample count
- Expected output sample count
- Actual output sample count  
- Conversion success rate
- Sample preview

## Code Structure

The refactored code follows these principles:

### Main Components
1. **ConversionConfig**: Data class for configuration management
2. **ChatMLConverter**: Main conversion class with type-specific methods
3. **Utility Functions**: File I/O, logging, and statistics calculation

### Key Methods
- `convert_dpo_sample()`: DPO-specific conversion logic
- `convert_sft_sample()`: SFT-specific conversion logic  
- `convert_cpt_sample()`: CPT-specific conversion logic
- `convert_dataset()`: Main conversion orchestration

## Testing

Test the converter with the provided sample files:

```bash
# Test DPO conversion
python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type dpo --reject-count 1

# Test SFT conversion  
python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type sft

# Test CPT conversion
python ./convert/chatml_converter.py ./data/test/HotpotEval_Corpus.json --convert-type cpt
```

## Troubleshooting

### Common Issues

1. **File Not Found**
   ```
   Error: Input file does not exist: ./data/nonexistent.json
   ```
   
   **Solution**: Check the file path and ensure the file exists

2. **Invalid JSON Format**
   ```
   Cannot load JSON file ./data/invalid.json: Expecting value: line 1 column 1 (char 0)
   ```
   
   **Solution**: Validate the JSON file format

3. **Permission Denied**
   ```
   Cannot save JSONL file ./data/train/output.jsonl: [Errno 13] Permission denied
   ```
   
   **Solution**: Check directory permissions

### Log Files

Detailed error information is available in the log files located in the `logs/` directory. Each conversion creates a timestamped log file with comprehensive debugging information.

## Contributing

When modifying the converter:

1. Follow the existing code structure and naming conventions
2. Add appropriate type hints and docstrings
3. Include comprehensive error handling
4. Update this documentation for any new features
5. Test all conversion types after changes

## License

This converter is part of the KG-DPO project. See the main project LICENSE for details.

---

For additional support, refer to the project documentation or create an issue in the project repository.