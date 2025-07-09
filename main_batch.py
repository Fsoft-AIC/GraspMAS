import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import warnings
import argparse
import torch
import asyncio
from pathlib import Path
from agents.graspmas import GraspMAS
from typing import List, DefaultDict, Dict, Any
import argparse
import json
import time
from pathlib import Path
from utils import eval_grasp, visualize_grasp_pose
from OCID_VLG.dataset import OCIDVLGDataset
from tqdm import tqdm

async def run_sample(grasp_mas: GraspMAS, sample: Dict, idx: int) -> Dict:
    print(f"Processing sample {idx}")
    query = sample["sentence"]
    image = sample["img"]
    start_time = time.time()
    try:
        result = await grasp_mas.query(query, image)
        status = "success"
    except Exception as e:
        result = str(e)
        status = "error"
    end_time = time.time()
    return {
        "index": idx,
        "query": query,
        "image": image,
        "status": status,
        "result": result,
        "elapsed_time_sec": round(end_time - start_time, 2)
    }

def chunk_dataset(dataset: Any, batch_size: int):
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        yield i, batch

async def evaluate(dataset, api_file: str, output_path: Path, max_round: int = 5, batch_size: int = 4):
    grasp_mas = GraspMAS(api_file=api_file, max_round=max_round)

    results = []
    for batch_start_idx, batch in chunk_dataset(dataset, batch_size):
        tasks = [
            run_sample(grasp_mas, sample, batch_start_idx + i)
            for i, sample in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete. Results saved to {output_path}")

    num_total = len(results)
    num_success = sum(1 for r in results if r['status'] == "success")
    num_error = num_total - num_success
    avg_time = sum(r['elapsed_time_sec'] for r in results) / num_total

    print("Evaluation Summary:")
    print(f"Total Samples  : {num_total}")
    print(f"✅ Success      : {num_success}")
    print(f"❌ Failures     : {num_error}")
    print(f"🎯 Success Rate : {num_success / num_total:.2%}")
    print(f"⏱️ Avg Time/Sample: {avg_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--api_file", type=str, required=True, help="Path to OpenAI API key file")
    parser.add_argument("--output", type=str, default="results/graspmas_eval_output.json", help="Output result file")
    parser.add_argument("--max_round", type=int, default=5, help="Max reasoning rounds")
    args = parser.parse_args()


    dataset = OCIDVLGDataset(args.dataset_path, split="test")

    asyncio.run(evaluate(dataset, args.api_file, Path(args.output), max_round=args.max_round))

if __name__ == "__main__":
    main()