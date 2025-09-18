'''
reference code from https://github.com/open-sciencelab/GraphGen
'''
import asyncio
import os
import gradio as gr

import json
from json_repair import repair_json
from tqdm.asyncio import tqdm as tqdm_async
import random
import re

from kg_dpo.configs import KGWalkingConfig
from kg_dpo.models import OpenAIModel, NetworkXStorage, Tokenizer, JsonKVStorage
from kg_dpo.templates import MULTI_HOP_GENERATION_PROMPT, NORMAL_QA_GENERATION_PROMPT
from kg_dpo.utils import detect_main_language, compute_content_hash, logger
from kg_dpo.operators.split_graph import get_batches_with_strategy

jsonl_save_file_path = os.getenv("JSONL_SAVE_FILE_PATH", "dpo.jsonl")

async def _pre_tokenize(graph_storage: NetworkXStorage,
                        tokenizer: Tokenizer,
                        edges: list,
                        nodes: list) -> tuple:

    sem = asyncio.Semaphore(1000)
    async def handle_edge(edge: tuple) -> tuple:
        async with sem:
            if 'length' not in edge[2]:
                edge[2]['length'] = len(
                    await asyncio.get_event_loop().run_in_executor(None,
                                                                   tokenizer.encode_string,
                                                                   edge[2]['description']))
            return edge

    async def handle_node(node: dict) -> dict:
        async with sem:
            if 'length' not in node[1]:
                node[1]['length'] = len(
                    await asyncio.get_event_loop().run_in_executor(None,
                                                                   tokenizer.encode_string,
                                                                   node[1]['description']))
            return node

    new_edges = []
    new_nodes = []

    for result in tqdm_async(asyncio.as_completed([handle_edge(edge) for edge in edges]),
                             total=len(edges), desc="Pre-tokenizing edges"):
        new_edge = await result
        await graph_storage.update_edge(new_edge[0], new_edge[1], new_edge[2])
        new_edges.append(new_edge)

    for result in tqdm_async(asyncio.as_completed([handle_node(node) for node in nodes]),
                             total=len(nodes), desc="Pre-tokenizing nodes"):
        new_node = await result
        await graph_storage.update_node(new_node[0], new_node[1])
        new_nodes.append(new_node)

    await graph_storage.index_done_callback()
    return new_edges, new_nodes

def extract_json(text: str):
    """
    从任意字符串中提取所有最外层 {...} 形式、合法的 JSON 对象。
    返回 list[dict]；无匹配时返回空 list。
    兼容 Python 3.7+ 标准库 re。
    """
    objs = []
    start = depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        # 处理转义，确保花括号出现在字符串里时不被计数
        if not escape and ch == '\\':
            escape = True
            continue
        if escape:
            escape = False
            continue

        if ch == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0:          # 最外层闭合
                try:
                    return text[start:i+1]
                except json.JSONDecodeError:
                    pass            # 跳过不合法片段

async def walking_graph_for_multi_hop(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    walking_strategy: KGWalkingConfig,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000
) -> dict:
    """
    Walking the graph for multi-hop

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param WalkingStrategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """
    assert walking_strategy.QA_FORM == "multi_hop"

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes,
        edges,
        graph_storage,
        walking_strategy
    )

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(_process_batch[0][0]['description']) == "zh" else "English"

                _process_nodes = _process_batch[0]
                _process_edges = _process_batch[1]

                entities = [
                    f"{_process_node['node_id']}: {_process_node['description']}" for _process_node in _process_nodes
                ]

                relations = [
                    f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
                    for _process_edge in _process_edges
                ]

                entities_str = "\n".join([f"{index + 1}. {entity}" for index, entity in enumerate(entities)])
                relations_str = "\n".join([f"{index + 1}. {relation}" for index, relation in enumerate(relations)])

                prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
                    entities=entities_str,
                    relationships=relations_str
                )

                context = await llm_client.generate_answer(prompt)

                # post-process the context
                if "Question:" in context and "Answer:" in context:
                    question = context.split("Question:")[1].split("Answer:")[0].strip()
                    answer = context.split("Answer:")[1].strip()
                elif "问题：" in context and "答案：" in context:
                    question = context.split("问题：")[1].split("答案：")[0].strip()
                    answer = context.split("答案：")[1].strip()
                else:
                    print("error format from model response...")
                    return {}, {}

                question = question.strip("\"")
                answer = answer.strip("\"")
                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)
               
                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": answer,
                        "entities": entities_str,
                        "relationships": relations_str
                    }
                }

            except Exception as e: # pylint: disable=broad-except
                print("Error occurred while processing batch: %s", e)
                logger.error("Error occurred while processing batch: %s", e)
                return {}, {}

    async for result in tqdm_async(
        asyncio.as_completed([_process_single_batch(batch) for batch in processing_batches]),
        total=len(processing_batches),
        desc="[2/3]Generating QAs"
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(processing_batches), desc="[2/3]Generating QAs")
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[2/3]Generating QAs")
        except Exception as e: # pylint: disable=broad-except
            # import traceback
            # traceback.print_exc()
            logger.error("Error occurred while generating QA: %s", e)
    return results

async def walking_graph_for_normal_qa(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    walking_strategy: KGWalkingConfig,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000
) -> dict:
    """
    Walking the graph for normal qa

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param walking_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """
    assert walking_strategy.QA_FORM == "normal_qa"

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes,
        edges,
        graph_storage,
        walking_strategy
    )

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(_process_batch[0][0]['description']) == "zh" else "English"

                _process_nodes = _process_batch[0]
                _process_edges = _process_batch[1]

                entities = [
                    f"{_process_node['node_id']}: {_process_node['description']}" for _process_node in _process_nodes
                ]

                relations = [
                    f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
                    for _process_edge in _process_edges
                ]

                entities_str = "\n".join([f"{index + 1}. {entity}" for index, entity in enumerate(entities)])
                relations_str = "\n".join([f"{index + 1}. {relation}" for index, relation in enumerate(relations)])

                prompt = NORMAL_QA_GENERATION_PROMPT[language].format(
                    number=5,
                    entities=entities_str,
                    relationships=relations_str
                )

                context = await llm_client.generate_answer(prompt)
                json_pattern = extract_json(context)
                qa_pairs = repair_json(json_pattern, return_objects=True, skip_json_loads=True, ensure_ascii=False)
                random_question, random_answer = random.choice(list(qa_pairs.items()))

                question = random_question.strip("\"")
                answer = random_answer.strip("\"")
                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)

                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": answer,
                        "entities": entities_str,
                        "relationships": relations_str                 
                        }
                }

            except Exception as e: # pylint: disable=broad-except
                print("Error occurred while processing batch: %s", e)
                logger.error("Error occurred while processing batch: %s", e)
                return {}, {}

    async for result in tqdm_async(
        asyncio.as_completed([_process_single_batch(batch) for batch in processing_batches]),
        total=len(processing_batches),
        desc="[2/3]Generating QAs"
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(processing_batches), desc="[2/3]Generating QAs")
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[2/3]Generating QAs")
        except Exception as e: # pylint: disable=broad-except
            # import traceback
            # traceback.print_exc()
            logger.error("Error occurred while generating QA: %s", e)
    return results
