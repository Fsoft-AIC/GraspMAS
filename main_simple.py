from agents.graspmas import GraspMAS
from argparse import ArgumentParser
import asyncio

def parse():
    args = ArgumentParser()
    args.add_argument("--api-file", type=str, default='api.key')
    args.add_argument("--query", type=str, required=True)
    args.add_argument("--image-path", type=str, required=True)
    args.add_argument("--max-round", type=int, default=4)
    return args.parse_args()

if __name__ == "__main__":
    print("Start!")
    args = parse()
    graspmas = GraspMAS(api_file=args.api_file, max_round=args.max_round)
    
    save_path, grasp_pose = asyncio.run(graspmas.query(args.query, args.image_path))
    print("End!")