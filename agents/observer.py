import os
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image
from .llm import OpenAILLM
from prompt import observer_prompt
from pathlib import Path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Observer:
    def __init__(self, observer_prompt: observer_prompt, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.observer_prompt = observer_prompt
        self.plan = None
        self.thought = None
        self.llm = llm

    async def __call__(self, execute_results: Union[str, Path], image_path: Path, **kwargs) -> Any:
        if isinstance(execute_results, str):
            observer_prompt1 = self.observer_prompt.OBSERVER.format(results=execute_results)
            base64_image = encode_image(image_path)
        elif isinstance(execute_results, Path):
            observer_prompt1 = observer_prompt.OBSERVER.format(results=None)
            base64_image = encode_image('imgs/grasp_pose_visualization.png')
        else:
            observer_prompt1 = observer_prompt.OBSERVER.format(results=None)
            base64_image = encode_image(image_path)

        content_message = [
            {
                "type": "text",
                "text": observer_prompt1,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                            },
            },
        ]
        

        input_message_observer = [
            {"role": "system", "content": self.llm.system_prompt},
            {
                "role": "user",
                "content": content_message,
            },
        ]

        response = await self.llm.client.chat.completions.create(
            model=self.model_name,
            messages=input_message_observer,
            max_tokens=self.max_tokens,
            stop=["\"\"\""]
        )
        observation = response.choices[0].message.content.split("<observation>")[1].split("</observation>")[0].strip()
        return observation