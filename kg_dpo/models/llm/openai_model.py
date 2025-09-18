import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from kg_dpo.models.llm.topk_token_model import TopkTokenModel, Token
from kg_dpo.models.llm.tokenizer import Tokenizer

def get_top_response_tokens(response: openai.ChatCompletion) -> List[Token]:
    token_logprobs = response.choices[0].logprobs.content
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob.logprob)
        candidate_tokens = [
            Token(t.token, math.exp(t.logprob))
            for t in token_prob.top_logprobs
        ]
        token = Token(token_prob.token, prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens

@dataclass
class OpenAIModel(TopkTokenModel):
    model_name: str = "Qwen3-30B-A3B"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0
    max_tokens: int = 25600

    system_prompt: str = ""
    json_mode: bool = False
    seed: int = None

    enable_think: bool = True
    token_usage: list = field(default_factory=list)


    def __post_init__(self):
        assert self.api_key is not None, "Please provide api key to access openai api."
        if self.api_key == "":
            self.api_key = "none"
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _pre_generate(self, text: str, history: List[str]) -> Dict:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": self.max_tokens,
            "extra_body": {"enable_thinking":self.enable_think}
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            assert len(history) % 2 == 0, "History should have even number of elements."
            messages = history + messages

        kwargs['messages']= messages
        return kwargs

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    async def generate_answer(self, text: str, history: Optional[List[str]] = None, temperature: int = 0) -> str:
        kwargs = self._pre_generate(text, history)
        kwargs["temperature"] = temperature
        

        prompt_tokens = 0
        for message in kwargs['messages']:
            prompt_tokens += len(Tokenizer().encode_string(message['content']))
        estimated_tokens = prompt_tokens + kwargs['max_tokens']

        completion = await self.client.chat.completions.create( # pylint: disable=E1125
            model=self.model_name,
            **kwargs
        )
        if hasattr(completion, "usage"):
            self.token_usage.append({
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            })
        answer_w_think = completion.choices[0].message.content
        answer = answer_w_think.split("</think>")[-1].strip() if self.enable_think else answer_w_think
        return answer

    async def generate_inputs_prob(self, text: str, history: Optional[List[str]] = None) -> List[Token]:
        raise NotImplementedError