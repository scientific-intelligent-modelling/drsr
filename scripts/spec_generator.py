#!/usr/bin/env python3
"""
生成可被当前管线识别的 specification（spec）文件：
- 支持自定义自变量名、因变量名、背景与问题描述
- 写入路径：results/drsr_{problem}_{timestamp}/specs/spec_{problem}_dynamic.txt

用法示例：
  python scripts/spec_generator.py \
    --problem oscillator1 \
    --n-features 2 \
    --feature-names x,v \
    --dependent a \
    --background "阻尼非线性振子，存在外驱动项；加速度受位置与速度影响，近似 a≈-k·x - c·v + f(t)"
"""

import argparse
import os
import time
from typing import List, Optional

DEFAULT_BACKGROUND = "该方程的物理性质未知，需要根据残差经验进行分析。"
DEFAULT_PROBLEM = "a damped nonlinear oscillator system with driving force"

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


def _ensure_feature_names(n: int, names: Optional[List[str]]) -> List[str]:
    if names is None:
        return [f"x{i+1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
    return names


def render_spec_numpy(
    n_features: int,
    feature_names: Optional[List[str]] = None,
    dependent_name: Optional[str] = None,
    background: Optional[str] = None,
    problem: Optional[str] = None,
    max_nparams: int = 10,
) -> str:
    feats = _ensure_feature_names(n_features, feature_names)
    dep = (dependent_name or "y").strip()
    bg = (background or DEFAULT_BACKGROUND).strip()
    prob = (problem or DEFAULT_PROBLEM).strip()

    feature_sig = ", ".join([f"{name}: np.ndarray" for name in feats])
    feature_doc = ", ".join(feats)

    # 线性种子：最多取前三个特征 + 常数项
    idx_c = min(3, n_features)
    terms = [f"params[{i}]*{feats[i]}" for i in range(min(3, n_features))]
    terms.append(f"params[{idx_c}]")
    linear_seed = " + ".join(terms)

    return SPEC_TEMPLATE_NUMPY.format(
        PROBLEM=prob,
        BACKGROUND=bg,
        FEATURE_SIG=feature_sig,
        FEATURE_DOC=feature_doc,
        DEPENDENT=dep,
        MAX_NPARAMS=max_nparams,
        LINEAR_SEED=linear_seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="oscillator1")
    parser.add_argument("--n-features", type=int, default=2)
    parser.add_argument("--feature-names", type=str, default=None, help="逗号分隔，如: x,v 或 x1,x2,x3")
    parser.add_argument("--dependent", type=str, default=None)
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--max-nparams", type=int, default=10)
    args = parser.parse_args()

    feats = None
    if args.feature_names:
        feats = [s.strip() for s in args.feature_names.split(",") if s.strip()]

    spec_text = render_spec_numpy(
        n_features=args.n_features,
        feature_names=feats,
        dependent_name=args.dependent,
        background=args.background,
        problem=f"acceleration in a {args.problem}" if args.problem == "oscillator1" else args.problem,
        max_nparams=args.max_nparams,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"drsr_{args.problem}_{ts}")
    out_dir = os.path.join(results_dir, "specs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spec_{args.problem}_dynamic.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(spec_text)

    print(f"[OK] Spec written: {out_path}")


if __name__ == "__main__":
    main()
