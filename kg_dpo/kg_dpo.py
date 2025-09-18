import asyncio
import os
import time
from dataclasses import dataclass
from typing import List, Union, cast
from tqdm.asyncio import tqdm as tqdm_async

import gradio as gr

from .configs import GLOBAL_CONFIG
from .models import (
    Chunk,
    JsonKVStorage,
    NetworkXStorage,
    OpenAIModel,
    Tokenizer
)
from .models.storage.base_storage import StorageNameSpace
from .operators import (
    extract_kg,
    walking_graph_for_multi_hop,
    walking_graph_for_normal_qa,
    ne_mining_with_none_context,
    ne_mining_with_corpus,
    ne_mining_with_subgraph
)
from .utils import compute_content_hash, create_event_loop, logger

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class KG_DPO:
    unique_id: str = str(int(time.time()))
    working_dir: str = os.path.join(sys_path, "cache")
    tokenizer_instance: Tokenizer = None
    llm_as_extractor_client: OpenAIModel = None
    llm_as_generator_client: OpenAIModel = None
    llm_as_miner_client: OpenAIModel = None
    kg_walking_strategy = GLOBAL_CONFIG.KG_WALKING
    ne_mining_strategy = GLOBAL_CONFIG.NE_MINING
    # webui
    progress_bar: gr.Progress = None

    def __post_init__(self):
        self.full_docs_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", self.unique_id), namespace="full_docs"
        )
        self.text_chunks_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", self.unique_id), namespace="text_chunks"
        )
        self.graph_storage: NetworkXStorage = NetworkXStorage(
            os.path.join(self.working_dir, "data", self.unique_id), namespace="graph"
        )
        self.qa_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", self.unique_id), namespace="qa"
        )
        self.dpo_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", self.unique_id), namespace="dpo"
        )

    async def async_split_chunks(self, data: Union[List[list], List[dict]]) -> dict:
        if len(data) == 0:
            return {}

        new_docs = {}
        inserting_chunks = {}
        assert isinstance(data, list) and isinstance(data[0], list)
        new_docs = {
            compute_content_hash("".join(chunk['content']), prefix="doc-"): {'content': "".join(chunk['content'])}
            for doc in data for chunk in doc
        }
        _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if len(new_docs) == 0:
            logger.warning("All docs are already in the storage")
            return {}
        logger.info("[New Docs] inserting %d docs", len(new_docs))
        async for doc in tqdm_async(data, desc="Chunking documents", unit="doc"):
            doc_str = "".join([chunk['content'] for chunk in doc])
            for chunk in doc:
                chunk_key = compute_content_hash(chunk['content'], prefix="chunk-")
                inserting_chunks[chunk_key] = {
                    **chunk,
                    'full_doc_id': compute_content_hash(doc_str, prefix="doc-")
                }
        _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}

        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)

        return inserting_chunks

    def insert(self, data: Union[List[list], List[dict]]):
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert(data))

    async def async_insert(self, data: Union[List[list], List[dict]]):
        """
        insert chunks into the graph
        """

        inserting_chunks = await self.async_split_chunks(data)

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return
        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))

        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.llm_as_extractor_client,
            kg_instance=self.graph_storage,
            tokenizer_instance=self.tokenizer_instance,
            chunks=[Chunk(id=k, content=v['content']) for k, v in inserting_chunks.items()],
            progress_bar = self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_instance in [self.full_docs_storage, self.text_chunks_storage,
                                 self.graph_storage, self.qa_storage, self.dpo_storage]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    def walking(self):
        if not self.qa_storage.data:
            loop = create_event_loop()
            loop.run_until_complete(self.async_walking())
        else:
            return

    async def async_walking(self):
        if self.kg_walking_strategy.QA_FORM == "multi_hop":
            results = await walking_graph_for_multi_hop(self.llm_as_generator_client,
                                                            self.tokenizer_instance,
                                                            self.graph_storage,
                                                            self.kg_walking_strategy,
                                                            self.text_chunks_storage,
                                                            self.progress_bar)
        elif self.kg_walking_strategy.QA_FORM == "normal_qa":
            results = await walking_graph_for_normal_qa(self.llm_as_generator_client,
                                                                self.tokenizer_instance,
                                                                self.graph_storage,
                                                                self.kg_walking_strategy,
                                                                self.text_chunks_storage,
                                                                self.progress_bar)
        else:
            raise ValueError(f"Unknown qa_form: {self.kg_walking_strategy.QA_FORM}")
        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    def mining(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_mining())

    async def async_mining(self):
        self.ne_mining_strategy.EMBEDDING_DB_PATH = os.path.join(self.working_dir, "data", self.unique_id, "faiss_index")

        if self.ne_mining_strategy.STRATEGY == "none_context":
            results = await ne_mining_with_none_context(self.llm_as_miner_client,
                                                             self.full_docs_storage,
                                                             self.qa_storage,
                                                             self.ne_mining_strategy,
                                                             self.progress_bar)
        elif self.ne_mining_strategy.STRATEGY == "retrieve_corpus":
            results = await ne_mining_with_corpus(self.llm_as_miner_client,
                                                             self.full_docs_storage,
                                                             self.qa_storage,
                                                             self.ne_mining_strategy,
                                                             self.progress_bar)
        elif self.ne_mining_strategy.STRATEGY == "subgraph":
            results = await ne_mining_with_subgraph(self.llm_as_miner_client,
                                                             self.full_docs_storage,
                                                             self.qa_storage,
                                                             self.ne_mining_strategy,
                                                             self.progress_bar)

        else:
            raise ValueError(f"Unknown ne_mining strategy: {self.ne_mining_strategy.STRATEGY}")
        await self.dpo_storage.upsert(results)
        await self.dpo_storage.index_done_callback()

    def clear(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_clear())

    async def async_clear(self):
        await self.full_docs_storage.drop()
        await self.text_chunks_storage.drop()
        await self.graph_storage.clear()
        await self.qa_storage.drop()
        await self.dpo_storage.drop()

        logger.info("All caches are cleared")
