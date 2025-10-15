import os
import time
from .llm import OpenAILLM
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
import cv2
import base64
from PIL import Image
from image_patch import ImagePatch
from pathlib import Path
from .coder import Coder
from .observer import Observer
from .planner import Planner
from grasp.utils import *
from agents.prompt import observer_prompt, planner_prompt, coder_prompt
from .llm import OpenAILLM
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class GraspMAS:
    def __init__(self, api_file, max_round=5):
        super().__init__()
        llm = OpenAILLM(api_file)
        self.planner = Planner(planner_prompt, llm)
        self.coder = Coder(coder_prompt, llm)
        self.observer = Observer(observer_prompt, llm)
        self.plan = None
        self.observation = None
        self.code = None
        self.max_round = max_round

    @staticmethod
    def _prepare_image(
        image: Union[str, np.ndarray], save_folder: Union[str, Path] = "imgs/"
    ) -> Tuple[np.ndarray, Path]:
        """Convert input (path or numpy array) to RGB numpy image and save a copy."""
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        image_path = save_folder / "query_image.png"

        if isinstance(image, str) or isinstance(image, Path):
            # load from file path
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return img, image_path

        if isinstance(image, np.ndarray):
            img = image.copy()
            cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return img, image_path

        raise ValueError("image must be str (path) or numpy.ndarray")

    async def query(
        self,
        query: str,
        image: Union[str, np.ndarray],
        save_folder: Union[str, Path] = "imgs/",
    ) -> Tuple[Any, Optional[List]]:
        """Run query through planner-coder-observer loop."""
        img_np, image_path = self._prepare_image(image, save_folder)

        grasp_pose: Optional[List] = None
        out: Any = None

        for idx in range(self.max_round):
            print("=" * 10, f"Round {idx}", "=" * 10)

            # Planner
            self.thought, self.plan = await self.planner(
                query=query, previous_plan=self.plan, observation=self.observation
            )
            print("----- Thought -----\n", self.thought)
            print("----- Plan -----\n", self.plan)

            if "return to user" in (self.plan or "").lower():
                break

            # Coder
            self.code = await self.coder(self.plan)
            print("----- Code -----\n", self.code)

            # Execute
            print("Executing code...")
            exec(self.code, globals())
            try:
                out_raw = execute_command(img_np)
                grasp_pose = out_raw.copy() if isinstance(out_raw, list) else None
            except Exception as e:
                out_raw = str(e)
                print("Error:", out_raw)

            result = {
                "grasp": None,
                "image": str(image_path),
                "error_logs": None,
            }
            
            # Visualization
            if isinstance(out_raw, list):
                result["grasp"] = out_raw
                grasp_pose = out_raw.copy()
                # save visualization
                vis_path = visualize_grasp_pose(img_np, out_raw, save_folder)
                print("Grasp Pose Visualization saved at:", vis_path)
                result["image"] = str(vis_path)
            else:
                result["error_logs"] = out_raw
            print("----- Execution Result -----\n", result)
            # Observer
            observation = await self.observer(result, query)
            print("----- Observation -----\n", observation)
            self.observation = observation['summary']
        return out, grasp_pose