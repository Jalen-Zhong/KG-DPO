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

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

from kg_dpo.configs import NEMiningConfig
from kg_dpo.models import OpenAIModel, JsonKVStorage
from kg_dpo.templates import DPO_GENERATION_PROMPT
from kg_dpo.utils import detect_main_language, compute_content_hash, logger
from kg_dpo.operators.split_graph import get_batches_with_strategy

async def ne_mining_with_none_context(
    llm_client: OpenAIModel,
    full_docs_storage: JsonKVStorage,
    qa_storage: JsonKVStorage,
    ne_mining_strategy: NEMiningConfig,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000
) -> dict:
    assert ne_mining_strategy.STRATEGY == "none_context"

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    qa_data = qa_storage.data.copy()
    questions = [v["question"] for v in qa_data.values()]
    answers = [v["answer"] for v in qa_data.values()]

    processing_batches = []
    for question, answer in zip(questions, answers):
        processing_batches.append((question, answer))

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(_process_batch[0]) == "zh" else "English"

                question = _process_batch[0]
                answer = _process_batch[1]
                negative_example_number = ne_mining_strategy.DEPTH

                context = ''

                prompt = DPO_GENERATION_PROMPT[language].format(
                    negative_example_number=negative_example_number,
                    question=question,
                    answer=answer,
                    context=context
                )

                output = await llm_client.generate_answer(prompt)
                reject_answers = repair_json(output, return_objects=True, skip_json_loads=True, ensure_ascii=False)

                return  {
                    compute_content_hash(question): {
                    "question": question,
                    "chosen": answer,
                    "rejected": reject_answers,
                    }
                }

            except Exception as e: # pylint: disable=broad-except
                # import traceback
                # traceback.print_exc()
                logger.error("Error occurred while processing batch: %s", e)
                return {}, {}

    async for result in tqdm_async(
        asyncio.as_completed([_process_single_batch(batch) for batch in processing_batches]),
        total=len(processing_batches),
        desc="[3/3]Generating DPO"
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(processing_batches), desc="[3/3]Generating DPO")
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[3/3]Generating DPO")
        except Exception as e: # pylint: disable=broad-except
            # import traceback
            # traceback.print_exc()
            logger.error("Error occurred while Generating DPO: %s", e)
    return results


async def ne_mining_with_corpus(
    llm_client: OpenAIModel,
    full_docs_storage: JsonKVStorage,
    qa_storage: JsonKVStorage,
    ne_mining_strategy: NEMiningConfig,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000
) -> dict:
    assert ne_mining_strategy.STRATEGY == "retrieve_corpus"

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    qa_data = qa_storage.data
    chunk_data = full_docs_storage.data

    questions = [v["question"] for v in qa_data.values()]
    answers = [v["answer"] for v in qa_data.values()]
    chunks = [v["content"] for v in chunk_data.values()]

    text_splitter = CharacterTextSplitter(chunk_size=ne_mining_strategy.CHUNK_SIZE, chunk_overlap=ne_mining_strategy.CHUNK_OVERLAP)
    docs = text_splitter.split_documents([Document(page_content=raw_text) for raw_text in chunks])
    embedding_model = OpenAIEmbeddings(
        model=ne_mining_strategy.EMBEDDING_MODEL_NAME,
        openai_api_base=ne_mining_strategy.EMBEDDING_API_BASE,
        openai_api_key=ne_mining_strategy.EMBEDDING_API_KEY
    )

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(ne_mining_strategy.EMBEDDING_DB_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": ne_mining_strategy.RETRIEVE_CORPUS_TOP_K})

    processing_batches = []
    for question, answer in zip(questions, answers):
        processing_batches.append((question, answer))

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(_process_batch[0]) == "zh" else "English"

                question = _process_batch[0]
                answer = _process_batch[1]
                negative_example_number = ne_mining_strategy.DEPTH

                question_corpus = retriever.invoke(question)
                question_corpus = [doc.page_content for doc in question_corpus]
                answer_corpus = retriever.invoke(answer)
                answer_corpus = [doc.page_content for doc in answer_corpus]
                corpus = set(question_corpus + answer_corpus)
                context = '```Corpus\n' + '\n'.join([f"{i+1}. {corpus}" for i, corpus in enumerate(corpus)]) + '\n```'
                prompt = DPO_GENERATION_PROMPT[language].format(
                    negative_example_number=negative_example_number,
                    question=question,
                    answer=answer,
                    context=context
                )
                output = await llm_client.generate_answer(prompt)
                reject_answers = repair_json(output, return_objects=True, skip_json_loads=True, ensure_ascii=False)

                return  {
                    compute_content_hash(question): {
                    "question": question,
                    "chosen": answer,
                    "rejected": reject_answers,
                    }
                }

            except Exception as e: # pylint: disable=broad-except
                import traceback
                traceback.print_exc()
                print("Error occurred while processing batch: %s", e)
                logger.error("Error occurred while processing batch: %s", e)
                return {}

    async for result in tqdm_async(
        asyncio.as_completed([_process_single_batch(batch) for batch in processing_batches]),
        total=len(processing_batches),
        desc="[3/3]Generating DPO"
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(processing_batches), desc="[3/3]Generating DPO")
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[3/3]Generating DPO")
        except Exception as e: # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)
    return results


async def ne_mining_with_subgraph(
    llm_client: OpenAIModel,
    full_docs_storage: JsonKVStorage,
    qa_storage: JsonKVStorage,
    ne_mining_strategy: NEMiningConfig,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000
) -> dict:
    assert ne_mining_strategy.STRATEGY == "subgraph"

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    qa_data = qa_storage.data
    questions = [v["question"] for v in qa_data.values()]
    answers = [v["answer"] for v in qa_data.values()]
    entities = [v["entities"] for v in qa_data.values()]
    relationships = [v["relationships"] for v in qa_data.values()]

    processing_batches = []
    for question, answer, entities, relationships in \
    zip(questions, answers, entities, relationships):
        processing_batches.append((question, answer, entities, relationships))

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(_process_batch[0]) == "zh" else "English"

                question = _process_batch[0]
                answer = _process_batch[1]
                entities = _process_batch[2]
                relationships = _process_batch[3]
                negative_example_number = ne_mining_strategy.DEPTH
                context = f'''```Entities\n{entities}\n```\n########\n```Relations\n{relationships}\n```'''

                prompt = DPO_GENERATION_PROMPT[language].format(
                    negative_example_number=negative_example_number,
                    question=question,
                    answer=answer,
                    context=context
                )
                output = await llm_client.generate_answer(prompt)
                reject_answers = repair_json(output, return_objects=True, skip_json_loads=True, ensure_ascii=False)

                return  {
                    compute_content_hash(question): {
                    "question": question,
                    "chosen": answer,
                    "rejected": reject_answers,
                    }
                }

            except Exception as e: # pylint: disable=broad-except
                # import traceback
                # traceback.print_exc()
                print("Error occurred while processing batch: %s", e)
                logger.error("Error occurred while processing batch: %s", e)
                return {}, {}

    async for result in tqdm_async(
        asyncio.as_completed([_process_single_batch(batch) for batch in processing_batches]),
        total=len(processing_batches),
        desc="[3/3]Generating DPO"
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(processing_batches), desc="[3/3]Generating DPO")
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[3/3]Generating DPO")
        except Exception as e: # pylint: disable=broad-except
            # import traceback
            # traceback.print_exc()
            logger.error("Error occurred while generating QA: %s", e)
    return results