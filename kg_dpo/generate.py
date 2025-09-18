import os
import json
import time
import argparse
import yaml
from typing import Optional
from pprint import pformat

from .kg_dpo import KG_DPO
from .models import OpenAIModel, Tokenizer
from .utils import set_logger
from .configs import GLOBAL_CONFIG, GlobalConfig

sys_path = os.path.abspath(os.path.dirname(__file__))

def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)

def load_config_from_yaml(config_path: str) -> GlobalConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Convert YAML data to GlobalConfig object
    return GlobalConfig(**config_data)

def pretty_print_config(config: GlobalConfig):
    """Pretty print configuration information"""
    print("=" * 60)
    print("KG-DPO Configuration Information")
    print("=" * 60)
    
    # Convert config object to dictionary
    config_dict = config.model_dump()
    
    # Use pformat for formatted output
    formatted_config = pformat(config_dict, width=100, indent=2, depth=3)
    print(formatted_config)
    print("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='KG-DPO Knowledge Graph Generation Tool')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path (.yaml/.yml)')
    parser.add_argument('--input-file', type=str, 
                       help='Input file path')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory')
    parser.add_argument('--cache-id', type=str, 
                       help='Cache ID')
    parser.add_argument('--extractor-model', type=str, 
                       help='Extractor model name')
    parser.add_argument('--generator-model', type=str, 
                       help='Generator model name')
    parser.add_argument('--miner-model', type=str, 
                       help='Miner model name')
    parser.add_argument('--tokenizer', type=str, 
                       help='Tokenizer name')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed configuration information')
    
    return parser.parse_args()

def update_config_from_args(config: GlobalConfig, args) -> GlobalConfig:
    """Update configuration from command line arguments"""
    if args.input_file:
        config.GENERAL.INPUT_FILE = args.input_file
    if args.output_dir:
        config.GENERAL.OUTPUT_DIR = args.output_dir
    if args.cache_id:
        config.GENERAL.CACHE_ID = args.cache_id
    if args.tokenizer:
        config.GENERAL.TOKENIZER = args.tokenizer
    if args.extractor_model:
        config.EXTRACTOR.MODEL_NAME = args.extractor_model
    if args.generator_model:
        config.GENERATOR.MODEL_NAME = args.generator_model
    if args.miner_model:
        config.MINER.MODEL_NAME = args.miner_model
    
    return config

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        # Load configuration from YAML file
        config = load_config_from_yaml(args.config)
        print(f"✅ Configuration loaded from file: {args.config}")
    else:
        # Use default configuration
        config = GLOBAL_CONFIG
        print("ℹ️  Using default configuration")
    
    # Update configuration with command line arguments
    config = update_config_from_args(config, args)
    
    # If verbose mode is enabled, pretty print configuration
    if args.verbose:
        pretty_print_config(config)
    else:
        # Concise mode: show only key information
        print(f"📁 Input file: {config.GENERAL.INPUT_FILE}")
        print(f"📂 Output directory: {config.GENERAL.OUTPUT_DIR}")
        print(f"🔧 Extractor model: {config.EXTRACTOR.MODEL_NAME}")
        print(f"🔧 Generator model: {config.GENERATOR.MODEL_NAME}")
        print(f"🔧 Miner model: {config.MINER.MODEL_NAME}")
        print(f"🎛️  Extractor temperature: {config.EXTRACTOR.TEMPERATURE}")
        print(f"🎛️  Generator temperature: {config.GENERATOR.TEMPERATURE}")
        print(f"🎛️  Miner temperature: {config.MINER.TEMPERATURE}")
    
    working_dir: str = config.GENERAL.OUTPUT_DIR
    set_working_dir(folder=working_dir)
    unique_id: str =  config.GENERAL.CACHE_ID if config.GENERAL.CACHE_ID else str(int(time.time()))
    set_logger(log_file=os.path.join(working_dir, "logs", f"{unique_id}.log"), if_stream=False)

    input_file: str = config.GENERAL.INPUT_FILE
    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)

    llm_as_extractor_client = OpenAIModel(model_name=config.EXTRACTOR.MODEL_NAME,
                                 api_key=config.EXTRACTOR.API_KEY,
                                 base_url=config.EXTRACTOR.BASE_URL,
                                 temperature=config.EXTRACTOR.TEMPERATURE,
                                 max_tokens=config.EXTRACTOR.MAX_TOKENS,
                                 enable_think=config.EXTRACTOR.ENABLE_THINK)

    llm_as_generator_client = OpenAIModel(model_name=config.GENERATOR.MODEL_NAME,
                            api_key=config.GENERATOR.API_KEY,
                            base_url=config.GENERATOR.BASE_URL,
                            temperature=config.GENERATOR.TEMPERATURE,
                            max_tokens=config.GENERATOR.MAX_TOKENS,
                            enable_think=config.GENERATOR.ENABLE_THINK)
    
    llm_as_miner_client = OpenAIModel(model_name=config.MINER.MODEL_NAME,
                            api_key=config.MINER.API_KEY,
                            base_url=config.MINER.BASE_URL,
                            temperature=config.MINER.TEMPERATURE,
                            max_tokens=config.MINER.MAX_TOKENS,
                            enable_think=config.MINER.ENABLE_THINK)

    kg_dpo_gen = KG_DPO(
        unique_id=unique_id,
        working_dir=working_dir,
        llm_as_extractor_client = llm_as_extractor_client,
        llm_as_generator_client = llm_as_generator_client,
        llm_as_miner_client = llm_as_miner_client,
        tokenizer_instance=Tokenizer(
            model_name=config.GENERAL.TOKENIZER
        ),
    )

    kg_dpo_gen.insert(data)

    kg_dpo_gen.walking()

    kg_dpo_gen.mining()

if __name__ == '__main__':
    main()
