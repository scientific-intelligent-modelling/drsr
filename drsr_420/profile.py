"""轻量化实验记录：保留样本 JSON 输出，移除 TensorBoard 依赖。

Profiler 现在直接接收 results_root（实验根目录），
在 results_root/samples 下输出每个样本的 JSON 文件。
"""

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from drsr_420 import code_manipulation
# 移除对 TensorBoard 的依赖，避免安装额外包


class Profiler:
    def __init__(
        self,
        results_root: str | None = None,
        pkl_dir: str | None = None,
        max_log_nums: int | None = None,
    ):
        """
        Args:
            results_root: 实验根目录（samples JSON 将保存在此目录下的 samples 子目录）。
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._results_root = results_root or '.'
        # samples 输出目录：results_root/samples
        self._json_dir = os.path.join(self._results_root, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        # 仅保留 samples 目录下分数最高的前 K 个样本 JSON
        self._keep_top_k_samples: int = 10

        # 不再创建 TensorBoard 写入器
        self._writer = None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        """兼容旧接口：不再写入 TensorBoard。"""
        return

    def _write_json(self, programs: code_manipulation.Function):
        """
        写入 JSON 的逻辑改为：
        - 不再单独写当前样本文件
        - 而是基于 _all_sampled_functions 重新计算前 K 个最优样本，
          统一以 topXX_ 前缀写入，保证文件名与排名一致
        """
        try:
            self._prune_samples_dir_topk()
        except Exception:
            # 精简/重写失败不影响主流程
            pass

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _prune_samples_dir_topk(self):
        """
        仅保留分数最高的前 K 个样本，并按排名重写文件：
        - 文件名形如：top01_samples_{sample_order}.json
        - JSON 字段顺序：sample_order -> score -> function -> params
        """
        # 1. 收集所有已有样本的分数
        entries = []
        for order, func in self._all_sampled_functions.items():
            try:
                s = getattr(func, 'score', None)
                if isinstance(s, (int, float)):
                    entries.append((order, float(s), func))
            except Exception:
                continue
        if not entries:
            return

        # 2. 按分数从高到低排序，取前 K 个
        entries.sort(key=lambda x: x[1], reverse=True)
        top_k = entries[: max(1, int(self._keep_top_k_samples))]

        # 3. 清空 samples 目录下旧的 JSON 文件
        try:
            for name in os.listdir(self._json_dir):
                if not name.endswith('.json'):
                    continue
                try:
                    os.remove(os.path.join(self._json_dir, name))
                except Exception:
                    pass
        except Exception:
            pass

        # 4. 重新按排名写入 topXX_ 前缀的文件
        for rank, (order, score, func) in enumerate(top_k, start=1):
            sample_order = order if order is not None else 0
            function_str = str(func)

            # 按用户需求的字段顺序组织内容：
            # 先是 sample_order，然后是 score，然后是方程（function），最后是参数
            content = {
                'sample_order': sample_order,
                'score': score,
                'function': function_str,
            }
            # 如果存在优化参数，则追加 params
            try:
                if getattr(func, 'optimized_params', None) is not None:
                    content['params'] = list(func.optimized_params)
            except Exception:
                pass

            file_name = f'top{rank:02d}_samples_{sample_order}.json'
            path = os.path.join(self._json_dir, file_name)
            try:
                with open(path, 'w', encoding='utf-8') as json_file:
                    json.dump(content, json_file, ensure_ascii=False, indent=2)
            except Exception:
                # 单个样本写失败不影响其它样本
                continue
