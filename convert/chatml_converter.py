#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatML Format Converter - Refactored Version

Convert different data formats to ChatML format, supporting DPO, SFT, CPT conversion types.

Usage examples:
    python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type dpo --reject-count 1
    python ./convert/chatml_converter.py ./output/data/1758011355/qa.json --convert-type sft
    python ./convert/chatml_converter.py ./output/data/1758011355/full_docs.json --convert-type cpt
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm


@dataclass
class ConversionConfig:
    """Conversion configuration"""
    system_message: str = "You are a helpful assistant."
    reject_count: int = 5
    convert_type: str = "dpo"


class ChatMLConverter:
    """ChatML format converter"""
    
    def __init__(self, config: ConversionConfig):
        """Initialize converter"""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _create_chatml_messages(self, 
                               question: Optional[str] = None, 
                               answer: Optional[str] = None) -> List[Dict[str, str]]:
        """Create ChatML message format"""
        messages = []
        
        # Add system message
        if self.config.system_message:
            messages.append({"role": "system", "content": self.config.system_message})
        
        # Add user message (if question exists)
        if question:
            messages.append({"role": "user", "content": question})
        
        # Add assistant message (if answer exists)
        if answer:
            messages.append({"role": "assistant", "content": answer})
        
        return messages
    
    def _extract_content_from_sample(self, sample: Any) -> Optional[str]:
        """Extract content text from sample"""
        if isinstance(sample, dict):
            # Try common content fields first
            for field in ['content', 'chosen', 'text', 'answer', 'response', 'output']:
                if field in sample and isinstance(sample[field], str):
                    return sample[field]
        elif isinstance(sample, str):
            return sample
        elif isinstance(sample, list) and sample:
            # Handle nested list structure
            first_item = sample[0]
            if isinstance(first_item, dict):
                return self._extract_content_from_sample(first_item)
        
        self.logger.warning(f"Cannot extract content from sample: {type(sample)}")
        return None
    
    def _extract_question_from_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract question text from sample"""
        if isinstance(sample, dict):
            # Try common question fields
            for field in ['question', 'query', 'input', 'prompt']:
                if field in sample and isinstance(sample[field], str):
                    return sample[field]
        return None
    
    def _extract_chosen_from_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract chosen/positive response from sample"""
        if isinstance(sample, dict):
            # Try common chosen response fields
            for field in ['chosen', 'positive', 'answer', 'response', 'output']:
                if field in sample and isinstance(sample[field], str):
                    return sample[field]
        return None
    
    def _extract_rejected_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        """Extract rejected/negative responses from sample"""
        rejected_list = []
        if isinstance(sample, dict):
            # Try common rejected response fields
            for field in ['rejected', 'negative', 'rejected_responses']:
                if field in sample and isinstance(sample[field], list):
                    rejected_list.extend([
                        item for item in sample[field] 
                        if isinstance(item, str) and item.strip()
                    ])
        return rejected_list
    
    def convert_dpo_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert DPO sample"""
        try:
            question = self._extract_question_from_sample(sample)
            chosen = self._extract_chosen_from_sample(sample)
            rejected_list = self._extract_rejected_from_sample(sample)
            
            # Validate required fields
            if not all([question, chosen, rejected_list]):
                return []
            
            results = []
            
            # Generate one sample for each rejected answer
            for rejected in rejected_list[:self.config.reject_count]:
                if not isinstance(rejected, str) or not rejected.strip():
                    continue
                
                result = {
                    "messages": self._create_chatml_messages(question, chosen),
                    "rejected_response": rejected
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"DPO sample conversion failed: {e}")
            return []
    
    def convert_sft_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert SFT sample"""
        try:
            question = self._extract_question_from_sample(sample)
            chosen = self._extract_chosen_from_sample(sample)
            
            if not all([question, chosen]):
                return []
            
            result = {
                "messages": self._create_chatml_messages(question, chosen)
            }
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"SFT sample conversion failed: {e}")
            return []
    
    def convert_cpt_sample(self, sample: Any) -> List[Dict[str, Any]]:
        """Convert CPT sample"""
        try:
            content = self._extract_content_from_sample(sample)
            
            if not content or not isinstance(content, str) or not content.strip():
                return []
            
            result = {
                "messages": self._create_chatml_messages(answer=content)
            }
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"CPT sample conversion failed: {e}")
            return []
    
    def convert_dataset(self, data: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """Convert entire dataset"""
        all_results = []
        
        # Determine data iterator
        if isinstance(data, dict):
            data_iter = data.values()
        elif isinstance(data, list):
            data_iter = data
        else:
            self.logger.error(f"Unsupported data format: {type(data)}")
            return []
        
        # Select appropriate conversion method
        convert_method = {
            "dpo": self.convert_dpo_sample,
            "sft": self.convert_sft_sample,
            "cpt": self.convert_cpt_sample
        }.get(self.config.convert_type)
        
        if not convert_method:
            self.logger.error(f"Unsupported conversion type: {self.config.convert_type}")
            return []
        
        # Convert all samples
        for sample in tqdm(data_iter, desc="Conversion progress", unit="sample"):
            try:
                converted_samples = convert_method(sample)
                all_results.extend(converted_samples)
            except Exception as e:
                self.logger.warning(f"Sample conversion failed: {e}")
        
        return all_results


def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_json_data(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """Load JSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Cannot load JSON file {file_path}: {e}")


def save_jsonl_data(data: List[Dict[str, Any]], output_file: str):
    """Save JSONL data"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        raise ValueError(f"Cannot save JSONL file {output_file}: {e}")


def calculate_statistics(data: Union[Dict[str, Any], List[Any]], 
                         convert_type: str) -> Dict[str, Any]:
    """Calculate conversion statistics"""
    if isinstance(data, dict):
        data_for_stats = list(data.values())
    elif isinstance(data, list):
        data_for_stats = data
    else:
        data_for_stats = []
    
    if convert_type == "cpt":
        expected_count = len(data_for_stats)
    elif convert_type == "sft":
        # For SFT, expected count equals input samples (one output per input)
        expected_count = len(data_for_stats)
    else:
        # For DPO, count rejected responses
        expected_count = 0
        for sample in data_for_stats:
            if isinstance(sample, dict):
                # Try multiple field names for rejected responses
                rejected_fields = ['rejected', 'negative', 'rejected_responses']
                for field in rejected_fields:
                    if field in sample and isinstance(sample[field], list):
                        expected_count += len(sample[field])
                        break
    
    return {
        "input_samples": len(data_for_stats),
        "expected_output": expected_count
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ChatML Format Converter - Convert data to ChatML format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_file', 
        help='Input JSON file path'
    )
    
    parser.add_argument(
        '--system-message', 
        default='You are a helpful assistant.',
        help='System message content'
    )
    
    parser.add_argument(
        '--convert-type', 
        type=str, 
        choices=['dpo', 'sft', 'cpt'],
        required=True,
        help='Conversion type: dpo, sft, cpt'
    )
    
    parser.add_argument(
        '--reject-count', 
        type=int, 
        default=5,
        help='Maximum number of rejected answers to keep for each DPO sample'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./data/train',
        help='Output file storage directory'
    )
    
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='logs',
        help='Log file storage directory'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        return 1
    
    # Set output file path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{args.convert_type}_chatml"
    if args.convert_type == 'dpo':
        output_filename += f"_depth_{args.reject_count}"
    output_filename += ".jsonl"
    
    output_file = Path(args.output_dir) / output_filename
    log_file = Path(args.log_dir) / f"{args.convert_type}_conversion_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(str(log_file))
    
    try:
        logger.info(f"Starting conversion: {args.input_file}")
        logger.info(f"Conversion type: {args.convert_type}")
        
        # Load data
        data = load_json_data(args.input_file)
        logger.info(f"Loaded samples: {len(data) if isinstance(data, list) else len(data.values())}")
        
        if (isinstance(data, list) and not data) or (isinstance(data, dict) and not data):
            logger.error("No data loaded")
            return 1
        
        # Calculate statistics
        stats = calculate_statistics(data, args.convert_type)
        
        # Configure converter
        config = ConversionConfig(
            system_message=args.system_message,
            reject_count=args.reject_count,
            convert_type=args.convert_type
        )
        
        converter = ChatMLConverter(config)
        
        # Convert data
        logger.info("Starting data conversion...")
        results = converter.convert_dataset(data)
        
        # Save results
        save_jsonl_data(results, str(output_file))
        
        # Output statistics
        success_rate = (len(results) / stats['expected_output'] * 100) if stats['expected_output'] > 0 else 0
        
        logger.info("=" * 60)
        logger.info("Conversion completed!")
        logger.info(f"Input samples: {stats['input_samples']}")
        logger.info(f"Expected output samples: {stats['expected_output']}")
        logger.info(f"Actual output samples: {len(results)}")
        logger.info(f"Conversion success rate: {success_rate:.2f}%")
        logger.info(f"Result file: {output_file}")
        logger.info(f"Log file: {log_file}")
        
        print(f"\n✅ Conversion completed!")
        print(f"   Input samples: {stats['input_samples']}")
        print(f"   Output samples: {len(results)}")
        print(f"   Success rate: {success_rate:.2f}%")
        print(f"   Result file: {output_file}")
        
        # Show sample example
        if results:
            print(f"\n📋 Sample example:")
            print(json.dumps(results[0], ensure_ascii=False, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        print(f"❌ Conversion failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())