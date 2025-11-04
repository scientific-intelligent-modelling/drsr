#!/usr/bin/env python3
"""
示例：使用自定义 X、y、变量名与背景，直接在内存渲染 specification 并启动管线。
不会写入 spec 文件；若需要文件式，请使用 scripts/spec_generator.py。
"""

import os
import time
import numpy as np
from typing import List, Optional

from drsr_420 import pipeline, config as config_lib, sampler, evaluator
from drsr_420 import prompt_config as pc

DEFAULT_BACKGROUND = "该方程的物理性质未知，需要根据残差经验进行分析。"
DEFAULT_PROBLEM = "acceleration in a damped nonlinear oscillator system with driving force"

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


def run_with_custom_xy(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]],
    dependent_name: Optional[str],
    background: Optional[str],
    problem: Optional[str] = None,
    max_sample_nums: int = 50,
):
    spec_text = render_spec_numpy(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=dependent_name,
        background=background,
        problem=problem,
    )

    dataset = {"data": {"inputs": X, "outputs": y}}
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_root = os.path.join("results", f"drsr_{(problem or 'custom')}_{ts}")
    logs_dir = os.path.join(results_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    cfg = config_lib.Config(use_api=False, api_model="gpt-3.5-turbo", results_root=results_root)
    class_cfg = config_lib.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)

    # 构造 PromptContext，保证提示文本与变量名/背景一致
    ctx = pc.PromptContext(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=dependent_name,
        problem_name=problem,
        background=background,
    )

    pipeline.main(
        specification=spec_text,
        inputs=dataset,
        config=cfg,
        max_sample_nums=max_sample_nums,
        class_config=class_cfg,
        log_dir=logs_dir,
        prompt_ctx=ctx,
    )


if __name__ == "__main__":
    # 示例数据：oscillator1（x, v -> a）
    rng = np.random.default_rng(42)
    N = 200
    x = rng.uniform(-1.0, 1.0, size=N)
    v = rng.uniform(-1.0, 1.0, size=N)
    a = -1.0 * x - 0.2 * v + rng.normal(0, 0.02, size=N)
    X = np.c_[x, v]
    y = a

    run_with_custom_xy(
        X, y,
        feature_names=["x", "v"],
        dependent_name="a",
        background="阻尼非线性振子，存在外驱动项；加速度受位置与速度影响，近似 a≈-k·x - c·v + f(t)",
        problem="oscillator1",
        max_sample_nums=10,
    )
