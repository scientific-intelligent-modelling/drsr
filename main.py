
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
parser.add_argument('--spec_path', type=str, default=None, help='可选：指定现有 spec 文件路径；不提供则使用 --data_csv 动态渲染')
parser.add_argument('--problem_name', type=str, default="problem")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--data_csv', type=str, default=None, help='当未提供 spec_path 时，使用该 CSV（含表头），前 n-1 列为特征，最后一列为因变量')
parser.add_argument('--background', type=str, default=None, help='背景知识（可选）')
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)

    # 构建统一的结果目录：results/{problem_name}_{timestamp}
    ts = time.strftime('%Y%m%d-%H%M%S')
    exp_name = f"{args.problem_name}_{ts}"
    results_root = os.path.join("results", exp_name)
    os.makedirs(results_root, exist_ok=True)
    # 所有产物直接放在实验根目录（不再创建 logs 子目录）

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

    # 不再设置独立日志目录

    config = config.Config(
        use_api=args.use_api,
        api_model=args.api_model,
        results_root=results_root,
    )
    # global_max_sample_num = 10000
    global_max_sample_num = 1000


    # ===============
    # 工具：从 CSV 读取数据与列名
    # ===============
    def _load_from_csv(path: str):
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError('CSV 至少需要两列（>=1 特征 + 1 因变量）')
        cols = list(df.columns)
        feature_names = cols[:-1]
        y_name = cols[-1]
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy().reshape(-1)
        return X, y, feature_names, y_name

    # ===============
    # 工具：渲染 specification（NumPy 版）
    # ===============
    DEFAULT_BACKGROUND = "The physical properties of this equation are unknown and need to be analyzed based on experience."
    DEFAULT_PROBLEM = args.problem_name

    SPEC_TEMPLATE_NUMPY = '''\
"""
Find the mathematical function skeleton that represents {PROBLEM}.

Background:
{BACKGROUND}

Variables:
- Independents: {FEATURE_DOC}
- Dependent: {DEPENDENT}
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = {MAX_NPARAMS}
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    else:
        return -loss_val

@equation.evolve
def equation({FEATURE_SIG}, params: np.ndarray) -> np.ndarray:
    """ Equation to be evolved.

    Background: {BACKGROUND}
    """
    # 初始线性骨架（可运行，便于快速启动；LLM 会逐步替换）
    return {LINEAR_SEED}
'''

    def _ensure_feature_names(n, names):
        if names is None:
            return [f"x{i+1}" for i in range(n)]
        if len(names) != n:
            raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
        return names

    def _render_spec_numpy(n_features, feature_names=None, dependent_name=None, background=None, problem=None, max_nparams=10):
        feats = _ensure_feature_names(n_features, feature_names)
        dep = (dependent_name or 'y').strip()
        bg = (background or DEFAULT_BACKGROUND).strip()
        prob = (problem or DEFAULT_PROBLEM).strip()
        feature_sig = ', '.join([f"{name}: np.ndarray" for name in feats])
        feature_doc = ', '.join(feats)
        idx_c = min(3, n_features)
        terms = [f"params[{i}]*{feats[i]}" for i in range(min(3, n_features))]
        terms.append(f"params[{idx_c}]")
        linear_seed = ' + '.join(terms)
        return SPEC_TEMPLATE_NUMPY.format(
            PROBLEM=prob,
            BACKGROUND=bg,
            FEATURE_SIG=feature_sig,
            FEATURE_DOC=feature_doc,
            DEPENDENT=dep,
            MAX_NPARAMS=max_nparams,
            LINEAR_SEED=linear_seed,
        )

    # 判断是否使用动态 CSV 接口
    use_dynamic = (args.spec_path is None) or (str(args.spec_path).strip() == '')

    if use_dynamic:
        # 1) 加载 CSV
        if not args.data_csv:
            raise SystemExit('未提供 spec_path 时，必须提供 --data_csv 指向含表头的 CSV 文件')
        X, y, feature_names, y_name = _load_from_csv(args.data_csv)
        background = args.background

        # 2) 渲染 specification
        specification = _render_spec_numpy(
            n_features=X.shape[1],
            feature_names=feature_names,
            dependent_name=y_name,
            background=background,
            problem=args.problem_name,
            max_nparams=10,
        )

        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}

        # 将动态渲染的 specification 保存到本次实验目录，便于调试
        try:
            spec_out_path = os.path.join(results_root, f"spec_{args.problem_name}_dynamic.txt")
            with open(spec_out_path, "w", encoding="utf-8") as f:
                f.write(specification)
            print(f"[INFO] Saved dynamic spec to: {spec_out_path}")
        except Exception as e:
            print(f"[WARN] Failed to save dynamic spec: {e}")
    else:
        # Load prompt specification（旧路径）
        with open(os.path.join(args.spec_path), encoding="utf-8") as f:
            specification = f.read()

        # Load dataset（旧数据路径）
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

        # 将使用的 specification 也复制一份到实验目录，便于调试
        try:
            import os as _os
            src_name = _os.path.basename(args.spec_path)
            spec_out_path = os.path.join(results_root, f"used_{src_name}")
            with open(spec_out_path, "w", encoding="utf-8") as f:
                f.write(specification)
            print(f"[INFO] Copied used spec to: {spec_out_path}")
        except Exception as e:
            print(f"[WARN] Failed to copy used spec: {e}")





##################################################
    # 定义 JSON 文件路径（放在实验根目录）
    json_experience_file = os.path.join(results_root, "experiences.json")

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
    
    
    # PromptContext：仅在动态模式下构造，保证提示一致性
    from drsr_420 import prompt_config as pc
    prompt_ctx = None
    if use_dynamic:
        prompt_ctx = pc.PromptContext(
            n_features=X.shape[1],
            feature_names=feature_names,
            dependent_name=y_name,
            problem_name=args.problem_name,
            background=background,
        )

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        results_root=results_root,
        prompt_ctx=prompt_ctx,
    )
