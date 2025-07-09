import os
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image
from .llm import OpenAILLM
from prompt import coder_prompt


class Coder:
    def __init__(self, coder_prompt: coder_prompt, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000,
                 temperature=0, top_p=1.0, frequency_penalty=0.0, presence_penalty=2.0):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.coder_prompt = coder_prompt
        self.example_coder = coder_prompt.EXAMPLES_CODER
        self.plan = None
        self.thought = None
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    async def __call__(self, plan: str, **kwargs) -> Any:
        code_prompt = self.coder_prompt.CODE.format(plan=plan, example=self.example_coder)

        input_message_coder = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": code_prompt},
        ]
        response = await self.llm.client.chat.completions.create(
            model=self.model_name,
            messages=input_message_coder,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=["\"\"\""])
        
        response_message_coder = response.choices[0].message.content
        code = '\n'.join(response_message_coder.split('\n')[1:-1]).replace('\n    \n', '\n')
        return code