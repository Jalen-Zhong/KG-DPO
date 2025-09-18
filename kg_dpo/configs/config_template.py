from pydantic import BaseModel, Field

class ExtractorConfig(BaseModel):
    MODEL_NAME: str = Field(default="Qwen3-30B-A3B", description="model name")
    BASE_URL: str = Field(default="https://YOUR_API_URL", description="base url")
    API_KEY: str = Field(default="YOUR_API_KEY", description="api key")
    TEMPERATURE: float = Field(default=0, ge=0, lt=2, description="temperature for sampling")
    MAX_TOKENS: int = Field(default=5120, gt=0, description="max tokens for sampling")
    ENABLE_THINK: bool = Field(default=True, description="if true, use LLM to think")

class GeneratorConfig(BaseModel):
    MODEL_NAME: str = Field(default="Qwen3-30B-A3B", description="model name")
    BASE_URL: str = Field(default="https://YOUR_API_URL", description="base url")
    API_KEY: str = Field(default="YOUR_API_KEY", description="api key")
    TEMPERATURE: float = Field(default=0.5, gt=0, lt=2, description="temperature for sampling")
    MAX_TOKENS: int = Field(default=5120, gt=0, description="max tokens for sampling")
    ENABLE_THINK: bool = Field(default=True, description="if true, use LLM to think")

class MinerConfig(BaseModel):
    MODEL_NAME: str = Field(default="Qwen3-30B-A3B", description="model name")
    BASE_URL: str = Field(default="https://YOUR_API_URL", description="base url")
    API_KEY: str = Field(default="YOUR_API_KEY", description="api key")    
    TEMPERATURE: float = Field(default=0.7, gt=0, lt=2, description="temperature for sampling")
    MAX_TOKENS: int = Field(default=12800, gt=0, description="max tokens for sampling")
    ENABLE_THINK: bool = Field(default=True, description="if true, use LLM to think")

class KGWalkingConfig(BaseModel):
    QA_FORM: str = Field(default="multi_hop", description="normal_qa, multi_hop")
    BIDRECTIONAL: bool = Field(default=True, description="if true, use bidirectional edges")
    ISOLATED_NODE_STRATEGY: str = Field(default="ignore", description="add, ignore")
    MAX_DEPTH: int = Field(default=2, gt=0)
    MAX_EXTRA_EDGES: int = Field(default=5, gt=0)
    MAX_TOKENS: int = Field(default=256, gt=0)
    EXAMPLE_NUM: int = Field(default=10, gt=0)

class NEMiningConfig(BaseModel):
    STRATEGY: str = Field(default="subgraph", description="none_context, retrieve_corpus, subgraph")
    DEPTH: int = Field(default=10, gt=0)
    EMBEDDING_MODEL_NAME: str = Field(default="Qwen3-Embedding-8B", description="model name")
    EMBEDDING_API_BASE: str = Field(default="https://YOUR_API_URL", description="embedding api base")
    EMBEDDING_API_KEY: str = Field(default="YOUR_API_KEY", description="embedding api key")
    CHUNK_SIZE: int = Field(default=4000, gt=0, description="chunk size for embedding")
    CHUNK_OVERLAP: int = Field(default=50, gt=0, description="chunk overlap for embedding")
    RETRIEVE_CORPUS_TOP_K: int = Field(default=2, gt=0, description="top k for retrieve corpus")
    EMBEDDING_DB_PATH: str = Field(default="output/faiss_index", description="embedding db path")

class GeneralConfig(BaseModel):
    TOKENIZER: str = Field(default="cl100k_base", description="tokenizer")
    INPUT_FILE: str = Field(default="./data/corpus/HotpotEval_Corpus.json", description="input file")
    OUTPUT_DIR: str = Field(default="output", description="output directory")
    CACHE_ID: str = Field(default="", description="unique id for cache")

class GlobalConfig(BaseModel):
    EXTRACTOR: ExtractorConfig = Field(default_factory=ExtractorConfig, description="config for extractor")
    GENERATOR: GeneratorConfig = Field(default_factory=GeneratorConfig, description="config for generator")
    MINER: MinerConfig = Field(default_factory=MinerConfig, description="config for miner")
    KG_WALKING: KGWalkingConfig = Field(default_factory=KGWalkingConfig, description="config for kg working")
    NE_MINING: NEMiningConfig = Field(default_factory=NEMiningConfig, description="config for ne mining")
    GENERAL: GeneralConfig = Field(default_factory=GeneralConfig, description="config for general")

GLOBAL_CONFIG: GlobalConfig = GlobalConfig()