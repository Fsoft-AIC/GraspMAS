import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from openai import AzureOpenAI, OpenAI, AsyncOpenAI
import base64

class OpenAILLM:
    def __init__(self, api_file):
        with open(api_file) as f:
            api_key = f.readline().splitlines()

        # self.client = OpenAI(
        #     api_key=api_key[0],
        # )
        self.client = AsyncOpenAI(api_key=api_key[0])
        self.system_prompt = "Please answer follows retrictly the provide format."
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
