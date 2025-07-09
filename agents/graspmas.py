import os
import time
from .llm import OpenAILLM
from prompt import planner_prompt_v4
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image
from image_patch import ImagePatch
from pathlib import Path
from .coder import Coder
from .observer import Observer
from .planner import Planner
from grasp.utils import *
from prompt import planner_prompt_v4, coder_prompt, observer_prompt
from .llm import OpenAILLM

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class GraspMAS:
    def __init__(self, api_file, max_round=5):
        super().__init__()
        llm = OpenAILLM(api_file)
        self.planner = Planner(planner_prompt_v4, llm)
        self.coder = Coder(coder_prompt, llm)
        self.observer = Observer(observer_prompt, llm)
        self.plan = None
        self.observation = None
        self.code = None
        self.max_round = max_round

    async def query(self, query: str, image: Union[str, Path, Image.Image, np.ndarray], save_folder='imgs/') -> Any:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image_path = 'imgs/query_image.png'
            image.save(image_path)
        elif isinstance(image, np.ndarray):
            image = image.copy()
            image_path = 'imgs/query_image.png'
            Image.fromarray(image).save(image_path)
        else:
            raise ValueError("image_path must be a str, Path, or PIL.Image")
        grasp_pose = None
        for idx in range(self.max_round):
            print(10*'=',f'Round {idx}',10*'=')
            ## Planner
            self.thought, self.plan = await self.planner(query=query, previous_plan=self.plan, observation=self.observation)
            print(5*'-',"Thought", 5*'-', '\n'+self.thought)
            print(5*'-',"Plan", 5*'-', '\n'+self.plan)
            if "return to user" in self.plan.lower():
                break
            ## Coder
            self.code = await self.coder(self.plan)
            print(5*'-',"Code", 5*'-', '\n' + self.code)
            ## Execute
            print("Executing code...")
            exec(self.code, globals())
            try:
                out = execute_command(image)
                grasp_pose = out.copy()
            except Exception as e:
                out = str(e)
                print("Error:", out)

            if isinstance(out, List):
                out = visualize_grasp_pose(np.array(image), out, save_folder)  
                print("Grasp Pose Visualization saved at:", out)
            ## Observer
            self.observation = await self.observer(out, image_path=image_path)
            print(5*'-',"Observation", 5*'-', '\n' + self.observation)

        return out, grasp_pose