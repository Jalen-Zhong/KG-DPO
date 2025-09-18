#!/usr/bin/env python3
"""
Multiple Choice Question (MCQ) Multi-Run Evaluation Script

This script performs multiple runs of multiple-choice question evaluation using OpenAI API,
calculates statistical metrics, and generates comprehensive reports with Excel files and boxplots.

Features:
- Multi-threaded evaluation for faster processing
- Statistical analysis of accuracy across multiple runs
- Excel report generation with detailed results
- Boxplot visualization of accuracy distributions
- Comprehensive logging and error handling

Example:
    python eval/evaluate_mcq_multi.py ./data/eval/Hotpot_test_mc.json --exp_name "hotpot-graphgen-dpo(subgraph_depth_1)-qwen2.5-1.5b-base"  --option_num 10 --num_runs 10
    python eval/evaluate_mcq_multi.py ./data/eval/PQA_test_mc.json --exp_name "pqa-graphgen-dpo(subgraph_depth_1)-qwen2.5-1.5b-base" --option_num 6 --num_runs 10
"""

import argparse
import json
import logging
import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor, as_completed


class MCQEvaluator:
    """Evaluator for multiple-choice questions using OpenAI API"""
    
    def __init__(self, base_url: str, api_key: str, model: str, option_num: int = 10):
        """
        Initialize the MCQ evaluator
        
        Args:
            base_url: OpenAI API base URL
            api_key: OpenAI API key
            model: Model name to use
            option_num: Number of options in multiple-choice questions
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.option_num = option_num
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def construct_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Construct prompt for multiple-choice question evaluation
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Formatted prompt string
        """
        question = question_data.get('question', '')
        options = question_data.get('options', [])
        
        prompt = f"""Please answer the following multiple-choice question. 

Question: {question}

Options:"""
        
        for i, option in enumerate(options[:self.option_num], 1):
            prompt += f"\n{i}. {option}"
        
        prompt += """

Please provide your answer as a single letter (A, B, C, etc.) corresponding to the correct option.
Answer: """
        
        return prompt
    
    async def get_model_response(self, prompt: str) -> str:
        """
        Get model response from OpenAI API
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Model response text
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 50
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
        except Exception as e:
            raise Exception(f"Failed to get model response: {str(e)}")
    
    def extract_answer(self, response: str) -> str:
        """
        Extract answer letter from model response
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer letter (A, B, C, etc.)
        """
        # Clean and extract the first letter from response
        response = response.strip().upper()
        
        # Look for the first occurrence of A-Z
        for char in response:
            if 'A' <= char <= 'Z':
                return char
        
        # If no letter found, return the first character
        return response[0] if response else 'X'
    
    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single multiple-choice question
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'question': question_data.get('question', ''),
            'options': question_data.get('options', []),
            'correct_answer': question_data.get('correct_answer', ''),
            'model_response': '',
            'extracted_answer': '',
            'is_correct': False,
            'error': None
        }
        
        try:
            prompt = self.construct_prompt(question_data)
            
            # Run async operation in sync context
            async def async_eval():
                async with self:
                    return await self.get_model_response(prompt)
            
            response = asyncio.run(async_eval())
            result['model_response'] = response
            
            extracted_answer = self.extract_answer(response)
            result['extracted_answer'] = extracted_answer
            
            # Check if answer is correct
            correct_answer = question_data.get('correct_answer', '').upper()
            result['is_correct'] = (extracted_answer == correct_answer)
            
        except Exception as e:
            result['error'] = str(e)
            result['is_correct'] = False
        
        return result


def setup_logging(log_file: str) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('mcq_evaluator')
    logger.setLevel(logging.WARNING)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.WARNING)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class MultiRunEvaluator:
    """Evaluator for running multiple evaluations and calculating statistics"""
    
    def __init__(self, evaluator: MCQEvaluator):
        """
        Initialize multi-run evaluator
        
        Args:
            evaluator: MCQEvaluator instance
        """
        self.evaluator = evaluator
        self.logger = setup_logging('multi_evaluation.log')
    
    def run_single_evaluation(self, data: List[Dict[str, Any]], max_workers: int = 64) -> Dict[str, Any]:
        """
        Run a single evaluation pass on all questions
        
        Args:
            data: List of question data
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_question = {
                executor.submit(self.evaluator.evaluate_single_question, q): q 
                for q in data
            }
            
            # Process completed tasks
            for future in as_completed(future_to_question):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    question = future_to_question[future]
                    self.logger.error(f"Error evaluating question: {question.get('question', 'Unknown')} - {str(e)}")
                    results.append({
                        'question': question.get('question', ''),
                        'error': str(e),
                        'is_correct': False
                    })
        
        # Calculate accuracy metrics
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        global_accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Calculate real accuracy (excluding errors)
        valid_results = [r for r in results if not r.get('error')]
        real_correct_count = sum(1 for r in valid_results if r.get('is_correct', False))
        real_total_count = len(valid_results)
        real_accuracy = real_correct_count / real_total_count if real_total_count > 0 else 0
        
        return {
            'results': results,
            'global_accuracy': global_accuracy,
            'real_accuracy': real_accuracy,
            'total_questions': total_count,
            'correct_count': correct_count,
            'error_count': total_count - len(valid_results)
        }
    
    def run_multiple_evaluations(self, data: List[Dict[str, Any]], num_runs: int = 10, max_workers: int = 64) -> Dict[str, Any]:
        """
        Run multiple evaluations and collect statistics
        
        Args:
            data: List of question data
            num_runs: Number of evaluation runs
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary with statistical results
        """
        global_accuracies = []
        real_accuracies = []
        all_results = []
        
        self.logger.warning(f"Starting {num_runs} evaluation runs...")
        
        for run_id in range(1, num_runs + 1):
            self.logger.warning(f"Starting evaluation run {run_id}/{num_runs}")
            
            result = self.run_single_evaluation(data, max_workers)
            
            global_acc = result['global_accuracy']
            real_acc = result['real_accuracy']
            
            global_accuracies.append(global_acc)
            real_accuracies.append(real_acc)
            
            # Add run_id to each result
            for res in result['results']:
                res['run_id'] = run_id
            all_results.extend(result['results'])
            
            self.logger.warning(f"Run {run_id} completed - Global accuracy: {global_acc:.4f}, Real accuracy: {real_acc:.4f}")
        
        # Calculate statistics
        stats = {
            'num_runs': num_runs,
            'global_accuracies': global_accuracies,
            'real_accuracies': real_accuracies,
            'global_stats': {
                'mean': mean(global_accuracies),
                'std': stdev(global_accuracies) if len(global_accuracies) > 1 else 0,
                'min': min(global_accuracies),
                'max': max(global_accuracies)
            },
            'real_stats': {
                'mean': mean(real_accuracies),
                'std': stdev(real_accuracies) if len(real_accuracies) > 1 else 0,
                'min': min(real_accuracies),
                'max': max(real_accuracies)
            },
            'all_results': all_results
        }
        
        return stats
    
    def save_results_to_excel(self, stats: Dict[str, Any], output_file: str):
        """
        Save results to Excel file with multiple sheets
        
        Args:
            stats: Statistical results dictionary
            output_file: Output Excel file path
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save statistical summary
            summary_data = {
                'Metric': ['Number of runs', 'Global accuracy mean', 'Global accuracy std', 'Global accuracy min', 'Global accuracy max',
                          'Real accuracy mean', 'Real accuracy std', 'Real accuracy min', 'Real accuracy max'],
                'Value': [
                    stats['num_runs'],
                    f"{stats['global_stats']['mean']:.4f}",
                    f"{stats['global_stats']['std']:.4f}",
                    f"{stats['global_stats']['min']:.4f}",
                    f"{stats['global_stats']['max']:.4f}",
                    f"{stats['real_stats']['mean']:.4f}",
                    f"{stats['real_stats']['std']:.4f}",
                    f"{stats['real_stats']['min']:.4f}",
                    f"{stats['real_stats']['max']:.4f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Statistical Summary', index=False)
            
            # Save accuracy details per run
            accuracy_data = {
                'Run Number': list(range(1, stats['num_runs'] + 1)),
                'Global Accuracy': stats['global_accuracies'],
                'Real Accuracy': stats['real_accuracies']
            }
            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_df.to_excel(writer, sheet_name='Accuracy Details', index=False)
            
            # Save detailed results
            detailed_df = pd.DataFrame(stats['all_results'])
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        self.logger.warning(f"Results saved to Excel file: {output_file}")
    
    def create_boxplot(self, stats: Dict[str, Any], output_file: str):
        """
        Create boxplot visualization of accuracy distributions
        
        Args:
            stats: Statistical results dictionary
            output_file: Output image file path
        """
        # Set font for proper text rendering
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Global accuracy boxplot
        ax1.boxplot(stats['global_accuracies'], labels=['Global Accuracy'])
        ax1.set_title('Global Accuracy Distribution')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical information
        global_mean = stats['global_stats']['mean']
        global_std = stats['global_stats']['std']
        ax1.text(0.02, 0.98, f'Mean: {global_mean:.4f}\nStd: {global_std:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Real accuracy boxplot
        ax2.boxplot(stats['real_accuracies'], labels=['Real Accuracy'])
        ax2.set_title('Real Accuracy Distribution')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical information
        real_mean = stats['real_stats']['mean']
        real_std = stats['real_stats']['std']
        ax2.text(0.02, 0.98, f'Mean: {real_mean:.4f}\nStd: {real_std:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.warning(f"Boxplot saved to: {output_file}")


def load_mcq_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load multiple-choice question data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of question data dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Multiple Choice Question Multi-Run Evaluation Tool')
    parser.add_argument('input_file', help='Path to input MCQ JSON file')
    parser.add_argument('--option_num', type=int, default=10, help='Number of options in multiple-choice questions')
    parser.add_argument('--base_url', default='https://ms-zt9zcnx4-100032905193-sw.gw.ap-beijing.ti.tencentcs.com/ms-zt9zcnx4/v1', 
                       help='OpenAI API base URL')
    parser.add_argument('--api_key', default='sk-abc123def0987111', help='OpenAI API key')
    parser.add_argument('--model', default='industry', help='Model name to use')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process')
    parser.add_argument('--max_workers', type=int, default=64, help='Maximum number of concurrent workers')
    parser.add_argument('--exp_name', type=str, default='hotpot-dpo-1.5b-instruct-full-none', help='Experiment name')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of evaluation runs')
    
    args = parser.parse_args()
    
    # Set up output file paths
    input_path = Path(args.input_file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    excel_file = input_path.parent / f"{input_path.stem}_{args.exp_name}_multi_results_{current_time}.xlsx"
    boxplot_file = input_path.parent / f"{input_path.stem}_{args.exp_name}_boxplot_{current_time}.png"
    log_file = f"logs/{input_path.stem}_multi_evaluation_{current_time}.log"
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Set up logging
    logger = setup_logging(str(log_file))
    
    try:
        # Load data
        logger.warning(f"Loading data file: {args.input_file}")
        data = load_mcq_data(args.input_file)
        
        # Limit processing if specified
        if args.limit:
            data = data[:args.limit]
        
        logger.warning(f"Loaded {len(data)} questions")
        logger.warning(f"Will perform {args.num_runs} evaluation runs")
        
        # Initialize evaluator
        evaluator = MCQEvaluator(args.base_url, args.api_key, args.model, args.option_num)
        multi_evaluator = MultiRunEvaluator(evaluator)
        
        # Run multiple evaluations
        stats = multi_evaluator.run_multiple_evaluations(
            data, args.num_runs, args.max_workers
        )
        
        # Save results to Excel
        multi_evaluator.save_results_to_excel(stats, str(excel_file))
        
        # Create boxplot
        multi_evaluator.create_boxplot(stats, str(boxplot_file))
        
        # Output statistical information
        logger.warning("="*60)
        logger.warning("Multi-run evaluation completed!")
        logger.warning(f"Number of runs: {stats['num_runs']}")
        logger.warning(f"Total questions: {len(data)}")
        logger.warning("\nGlobal accuracy statistics:")
        logger.warning(f"  Mean: {stats['global_stats']['mean']:.4f} ({stats['global_stats']['mean']*100:.2f}%)")
        logger.warning(f"  Std: {stats['global_stats']['std']:.4f}")
        logger.warning(f"  Min: {stats['global_stats']['min']:.4f} ({stats['global_stats']['min']*100:.2f}%)")
        logger.warning(f"  Max: {stats['global_stats']['max']:.4f} ({stats['global_stats']['max']*100:.2f}%)")
        logger.warning("\nReal accuracy statistics:")
        logger.warning(f"  Mean: {stats['real_stats']['mean']:.4f} ({stats['real_stats']['mean']*100:.2f}%)")
        logger.warning(f"  Std: {stats['real_stats']['std']:.4f}")
        logger.warning(f"  Min: {stats['real_stats']['min']:.4f} ({stats['real_stats']['min']*100:.2f}%)")
        logger.warning(f"  Max: {stats['real_stats']['max']:.4f} ({stats['real_stats']['max']*100:.2f}%)")
        logger.warning(f"\nExcel results file: {excel_file}")
        logger.warning(f"Boxplot file: {boxplot_file}")
        logger.warning(f"Log file: {log_file}")
        
        print(f"\nMulti-run evaluation completed!")
        print(f"Number of runs: {stats['num_runs']}")
        print(f"Global accuracy: {stats['global_stats']['mean']:.4f} ± {stats['global_stats']['std']:.4f}")
        print(f"Real accuracy: {stats['real_stats']['mean']:.4f} ± {stats['real_stats']['std']:.4f}")
        print(f"Results file: {excel_file}")
        print(f"Boxplot: {boxplot_file}")
        
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise


if __name__ == '__main__':
    main()