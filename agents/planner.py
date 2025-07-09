import os
import time
from .llm import OpenAILLM
from prompt import planner_prompt_v4
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image

class Planner:
    def __init__(self, planner_prompt: planner_prompt_v4, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.planner_prompt = planner_prompt
        self.example_planner = planner_prompt.EXAMPLES_PLANNER_1
        self.plan = None
        self.thought = None
        self.llm = llm
    
    async def __call__(self, query: str, previous_plan: str, observation: str, **kwargs) -> Any:
        planner_prompt = self.planner_prompt.PLAN.format(user_query=query, examples=self.example_planner, 
                                                            planning=previous_plan, observer_output=observation)

        input_message_planner = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": planner_prompt},
        ]
                
        response = await self.llm.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=input_message_planner,
            max_tokens=1000,
            stop=["\"\"\""])

        thought_plan = response.choices[0].message.content.strip()
        thought = thought_plan.split('<thought>')[1].split('</thought>')[0].strip()
        plan =  thought_plan.split('<plan>')[1].split('</plan>')[0].strip()

        return thought, plan
