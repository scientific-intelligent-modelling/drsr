
import json
import os
import time
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd
import sys

from drsr_420 import pipeline
from drsr_420 import config
from drsr_420 import sampler
from drsr_420 import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str,default="./specs/specification_oscillator1_numpy.txt")
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)

    # 构建统一的结果目录：results/{problem_name}_{timestamp}
    ts = time.strftime('%Y%m%d-%H%M%S')
    exp_name = f"{args.problem_name}_{ts}"
    results_root = os.path.join("results", exp_name)
    logs_dir = os.path.join(results_root, "logs")
    experiences_dir = os.path.join(results_root, "equation_experiences")
    residual_dir = os.path.join(results_root, "residual_analyze")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(experiences_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)

    # 将标准输出和错误输出同时写入结果目录，便于统一归档
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    out_path = os.path.join(results_root, "run.out")
    err_path = os.path.join(results_root, "run.err")
    out_fp = open(out_path, "a", encoding="utf-8")
    err_fp = open(err_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, out_fp)
    sys.stderr = _Tee(sys.stderr, err_fp)
    print(f"[INFO] Results root: {results_root}")

    # 统一覆盖日志目录到 results/{exp}/logs
    args.log_path = logs_dir

    config = config.Config(
        use_api=args.use_api,
        api_model=args.api_model,
        results_root=results_root,
    )
    # global_max_sample_num = 10000
    global_max_sample_num = 1000


    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 





##################################################
    # 创建经验保存目录
    experience_dir = experiences_dir
    os.makedirs(experience_dir, exist_ok=True)
    # 定义 JSON 文件路径
    json_experience_file = os.path.join(experience_dir, "experiences.json")

    # 如果文件不存在，则创建初始结构
    if not os.path.exists(json_experience_file):
        initial_experiences = {
            "None": [], 
            "Good": [], 
            "Bad": []
        }
        
        try:
            with open(json_experience_file, "w", encoding="utf-8") as f:
                json.dump(initial_experiences, f, ensure_ascii=False, indent=2)
            print(f"成功创建初始经验 JSON 文件: {json_experience_file}")
        except Exception as e:
            print(f"创建 JSON 文件时出错: {str(e)}")
    else:
        print(f"经验 JSON 文件已存在: {json_experience_file}")
##################################################
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        # log_dir = 'logs/m1jobs-mixtral-v10',
        log_dir=args.log_path,
    )
