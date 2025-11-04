# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

import random
from drsr_420 import evaluator
from drsr_420 import buffer
from drsr_420 import config as config_lib
import requests
import json
# http.client 不再使用，调用统一的 llm.ClientFactory
import os
import traceback
from typing import Any

# 简单清洗：保留到首个 return 为止，移除危险/无关行，并确保缩进
def _sanitize_equation_body(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()
    cleaned = []
    found_return = False
    blacklist = (
        'import ', 'print(', 'open(', 'os.', 'sys.', '__import__', 'subprocess', 'eval(', 'exec(', 'if __name__', 'while True:'
    )
    for raw in lines:
        s = raw.strip()
        if not s:
            # 跳过开头的空行
            if not cleaned:
                continue
        # 黑名单过滤
        lower_line = s.lower()
        if any(tok in lower_line for tok in blacklist):
            continue
        cleaned.append(('    ' + s) if s else s)  # 确保基本缩进
        if s.startswith('return'):
            found_return = True
            break
    # 若没有 return，尽量返回清理后的内容
    return "\n".join(cleaned) + ("\n" if cleaned else "")

SHARED_LLM_CLIENT: Any = None

def set_shared_llm_client(client):
    global SHARED_LLM_CLIENT
    SHARED_LLM_CLIENT = client
problem_name_in_prompt = 'a damped nonlinear oscillator system with driving force'
dependent_name_in_prompt = 'acceleration'
independent_name_in_prompt = 'position, and velocity'
Port = '5000'

# 采样与分析时的最大输出 token；模型名需由外部 config 显式提供
class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    # self._samples_per_prompt = 4 每一次prompt都生成四个相互独立的回答



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

# python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()    # 从岛上拿一个可参考的方程框架 - 故可以独立反思

            island_id = prompt.island_id

            best_score = self._database._best_score_per_island[island_id]
            # 静默化：不显示每个岛屿的prompt获取信息
            # print(f"从岛屿 {island_id} 获取prompt，最佳分数: {best_score}")

            reset_time = time.time()

            # 静默化：不显示LLM调用信息
            # print("调用大模型处理")

            # 01 版本
            # samples, sed_rep = self._llm.draw_samples(prompt.code,self.config) # 向大模型采样出一个方程框架 - 核心
            samples = self._llm.draw_samples(prompt.code,self.config) # 向大模型采样出一个方程框架 - 核心
  
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # 静默化：不显示采样结果
            # print("获得了samples，在95行")
            # print(samples)
            # This loop can be executed in parallel on remote evaluator machines.
            score_for_sample = []
            error_for_samlple = []
            opt_params_for_sample = []
            quality_for_sample = []
            residual_data = None  # 用于存储每个样本的残差数据
            best_sample = None
            if_best = False
            id = 0
            temp_best_score = []
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                # 先清洗样本，去除测试代码/危险语句，仅保留函数体片段
                sample_clean = _sanitize_equation_body(sample)
                score, error_msg, residual, opt_params = chosen_evaluator.analyse(
                    sample_clean,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                score_for_sample.append(score)
                error_for_samlple.append(error_msg)
                opt_params_for_sample.append(opt_params)
                id += 1
                # 静默化：不显示每个样本的评估细节
                # print(best_score)
                # print(score)
                # print('===================从chosen_evaluator.analyse中获得残差=====================\n')
                # print(residual)
                if score is not None and score > best_score:
                # if score is not None :#先为了调试，都搞一遍，上面的才是需要的
                    temp_best_score.append(score)
                    #如果score比temp_best_score中的最大值大，就更新best
                    if score >= max(temp_best_score):
                        best_id = id
                        if_best = True
                        # 静默化：不显示调试信息
                        # print("我在这里变成true了")
                        residual_data=residual
                        best_sample = sample
                        best_score_for_sample = score

                        # 添加最优值更新通知
                        self._notify_new_best_score(island_id, score, sample)
            # print("一共有多少个sample？",i)
           
            # 静默化：不显示详细的样本数据
            # print("score_for_sample: ")
            # print(score_for_sample)
            # print("===========error_for_samlple:============================\n ")
            # print(error_for_samlple)
            # print("=========================residual_data: ================\n")
            # print(residual_data)
            for each_score in score_for_sample:
                if each_score == None:
                    quality_for_sample.append('None')
                elif each_score > best_score:
                    quality_for_sample.append('Good')
                else:
                    quality_for_sample.append('Bad')
            # 静默化：不显示质量检查详情
            # print("quality_for_sample:")
            # print('================================检查一下if_best的值====================\n')
            # print(if_best)
            # 调用分析函数进行分析
            try:
                #先直接进入第三次
                # 静默化：不显示分析过程详情
                # print("\n===== 方程和分数分析开始 =====")
                # 使用清洗后的样本进行分析，提高一致性
                cleaned_samples = [ _sanitize_equation_body(s) for s in samples ]
                analysis_result = self.analyze_equations_with_scores(cleaned_samples, quality_for_sample, error_for_samlple, prompt)
                # print("总的分析结果：---------")
                # print(analysis_result)
                # print("===== 方程和分数分析结束 =====\n")

                # 添加第三次对话：残差分析
                # 静默化：不显示残差分析详情
                # print("\n===== 残差分析开始 =====")
                # print(residual_data)
                # print(if_best)
                if residual_data is not None and if_best:
                    # 只对有效样本进行残差分析
                    if_best = False
                    residual_result = self.analyze_equations_with_residual(best_sample,residual_data)
                    # 静默化：不显示残差分析结果
                    # print(f"样本残差分析结果: {residual_result}")
                    # 创建目录存放残差分析结果
                    residual_analyze_dir = os.path.join(self.config.results_root or ".", "residual_analyze")
                    if not os.path.exists(residual_analyze_dir):
                        os.makedirs(residual_analyze_dir)
                    
                    json_residual_file = os.path.join(residual_analyze_dir, "residual_analyze.json")
                    
                    # 加载现有的残差分析数据（如果文件存在）
                    residual_data_list = []
                    if os.path.exists(json_residual_file):
                        try:
                            with open(json_residual_file, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                                if isinstance(existing_data, list):
                                    residual_data_list = existing_data
                        except json.JSONDecodeError:
                            # 静默化：不显示文件格式错误信息
                           print(f"现有的残差分析JSON文件格式有误，将创建新文件")
                        except Exception as e:
                            # 静默化：简化错误输出
                            print(f"读取现有残差分析文件时出错: {e}")
                    
                    # 创建新的残差分析记录
                    current_sample_order = self._get_global_sample_nums() - len(samples) + best_id  # 获取当前样本的顺序号
                    
                    # 计算残差统计信息
                    res_values = residual_data[:, -1]  # 第三列是残差值

                    # 创建残差分析数据结构
                    residual_record = {
                        "sample_order": current_sample_order,
                        "island_id": prompt.island_id,
                        "equation": best_sample,
                        "analysis": residual_result,
                        "best_score": best_score_for_sample,
                    }
                    
                    # 添加到残差分析数据列表
                    residual_data_list.append(residual_record)
  
                    # 保存更新后的残差分析数据
                    try:
                        with open(json_residual_file, "w", encoding="utf-8") as f:
                            json.dump(residual_data_list, f, ensure_ascii=False, indent=2)
                        # 静默化：不显示文件操作成功信息
                        # print(f"成功更新残差分析JSON文件: {json_residual_file}")
                    except Exception as e:
                        # 保留错误信息，但简化输出
                        # print(f"保存残差分析JSON文件时出错: {e}")
                        pass

                # 静默化：不显示分析结束标记
                # print("===== 残差分析结束 =====\n")

                # 创建目录存放分析结果
                # import os
                experience_dir = os.path.join(self.config.results_root or ".", "equation_experiences")
                
                json_experience_file = os.path.join(experience_dir, "experiences.json")
                

                # 加载现有的经验（如果文件存在）
                experiences_data = {"None": [], "Good": [], "Bad": []}
                if os.path.exists(json_experience_file):
                    try:
                        with open(json_experience_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            # 确保键存在
                            for key in ["None", "Good", "Bad"]:
                                if key in existing_data:
                                    experiences_data[key] = existing_data[key]
                    except json.JSONDecodeError:
                        # 静默化：不显示文件格式错误信息
                        print(f"现有的 JSON 文件格式有误，将创建新文件")
                    except Exception as e:
                        # 静默化：简化错误输出
                        print(f"读取现有经验文件时出错: {e}")

                # 添加新经验
                for i, (sample_text, quality, analysis, error_msg) in enumerate(zip(cleaned_samples, quality_for_sample, analysis_result, error_for_samlple)):
                    # 获取当前样本的信息
                    current_sample_order = self._get_global_sample_nums() - len(samples) + i + 1  # 计算当前样本的顺序号
                    
                    # 确定分类
                    if quality == 'Good':
                        category = "Good"
                    elif quality == 'Bad':
                        category = "Bad"
                    else:  # 'None'
                        category = "None"
                    
                    # 创建经验数据结构
                    experience = {
                        "island_id": prompt.island_id,
                        "analysis": analysis,
                        "sample_order": current_sample_order,  # 添加样本顺序号
                        "sample_time": sample_time,
                        # 仅保存清洗后的函数体，避免测试代码/危险语句
                        "equation": sample_text,
                        "score": score_for_sample[i],
                    }
                    # 保存训练拟合参数
                    try:
                        params_i = opt_params_for_sample[i]
                        if params_i is not None:
                            import numpy as _np
                            experience["fitted_params"] = _np.asarray(params_i).tolist()
                    except Exception:
                        pass
                    
                    # 对于 None 类型，添加错误信息
                    if category == "None" and error_msg:
                        experience["error"] = error_msg
                    
                    # 添加到相应类别
                    experiences_data[category].append(experience)

                # 保存更新后的经验数据
                try:
                    with open(json_experience_file, "w", encoding="utf-8") as f:
                        json.dump(experiences_data, f, ensure_ascii=False, indent=2)
                    # 静默化：不显示文件操作成功信息
                    # print(f"成功更新经验 JSON 文件: {json_experience_file}")
                except Exception as e:
                    # 简化错误输出
                    # print(f"保存 JSON 经验文件时出错: {e}")
                    pass
            except Exception as e:
                # 静默化：简化错误输出
                # print(f"执行分析时出错: {str(e)}")
                pass

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1

    def _notify_new_best_score(self, island_id, score, sample):
        """通知新的最优分数"""
        try:
            # 提取方程的主要部分
            lines = sample.split('\n')
            main_equation = ""
            for line in lines:
                if 'dv =' in line or 'return' in line:
                    main_equation += line.strip() + "; "

            if not main_equation:
                main_equation = lines[0][:100] + "..." if len(lines) > 0 and len(lines[0]) > 100 else (lines[0] if lines else "Unknown")

            # 检查是否是全局最优
            current_global_best = max(self._database._best_score_per_island.values()) if self._database._best_score_per_island else float('-inf')

            if score > current_global_best:
                print(f"🏆 NEW GLOBAL BEST: Island {island_id}, Score = {score:.6f}")
                print(f"   Equation: {main_equation[:200]}{'...' if len(main_equation) > 200 else ''}")
            else:
                print(f"📈 Island {island_id} improved: Score = {score:.6f}")

        except Exception:
            # 如果提取方程失败，只显示基本信息
            print(f"📈 Island {island_id} improved: Score = {score:.6f}")



    # 02 版本 good/bad/none的分析
    def analyze_equations_with_scores(self, samples, quality_for_sample, error_for_sample, prompt):
        """
        让大模型分析方程及其得分，提供思考过程分析和改进建议
        
        Args:
            samples: 生成的方程样本列表
            scores: 对应的分数列表
            prompt: 原始提示，用于提供上下文
            
        Returns:
            analysis_result: 模型对方程的分析结果
        """

        analysis_results = []

        i = 0

        for sample_each in samples:

            if quality_for_sample[i] == 'Good':
                new__question = f"""
                The optimized function skeleton you just answered scored higher. Please summarize useful experience.
                STRICTLY follow these rules:
                1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents {dependent_name_in_prompt} in{problem_name_in_prompt}, I can ..."
                2. Summarize ONLY the key success factors
                3. You need to make your answer as concise as possible
                """
# 1. when seeking for the mathematical function skeleton that represents acceleration in{problem_name_in_prompt}, I can
# 2. Identify ONE crucial improvement point
# 3. You need to make your answer as concise as possible
            elif quality_for_sample[i] == 'Bad':
                new__question = f"""
                The optimized function skeleton you just answered scored lower. What lessons can you draw from it?
                STRICTLY follow these rules: 
                1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents {dependent_name_in_prompt} in{problem_name_in_prompt}, I can ..."
                2. Identify ONE crucial improvement point
                3. You need to make your answer as concise as possible
                """
            
            elif quality_for_sample[i] == 'None':
                new__question = f"""
                The optimized function skeleton you just answered failed with error: {error_for_sample[i]}, What lessons can you draw from it?
                STRICTLY follow these rules:
                1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents {dependent_name_in_prompt} in{problem_name_in_prompt}, I need ..."
                2. Address the SPECIFIC error: {error_for_sample[i]}
                3. Identify ONE crucial improvement point
                4. You need to make your answer as concise as possible
                """
            i += 1

            analysis_prompt = f"""
            Here's our previous conversation:

            user: {prompt}

            assistant: {sample_each}

            user: {new__question}
            """
            # 调用远程API分析结果
            try:
                client = SHARED_LLM_CLIENT
                if client is None:
                    raise ValueError("未注入共享 LLM 客户端，请在 Wrapper 中注入后再运行。")
                resp = client.chat([{"role": "user", "content": analysis_prompt}])
                analysis_result = resp.get('content', '') or ''
                # 静默化：不显示分析结果详情
                # print(f"分析结果：{analysis_result}")
                analysis_results.append(analysis_result or "分析为空")
            except Exception as e:
                # 静默化：简化错误输出
                # print(f"分析请求发生错误: {str(e)}")
                analysis_results.append(f"分析请求发生错误: {str(e)}")
        return analysis_results

    def analyze_equations_with_residual(self, sample, residual) -> str:
        """
    让大模型根据输入的残差分析方程，提供对数据的思考过程分析和改进建议
    
    Args:
        sample: 生成的方程样本
        residual: 输入的数据残差
        prompt: 原始提示，用于提供上下文
            
    Returns:
        analysis_result: 模型对方程的分析结果
        """
        # 静默化：不显示函数进入标记
       # print("========================进入了残差分析函数========================")
        # 直接使用传入的残差数据
        # 计算残差的统计信息
        res_values = residual[:, -1]  # 第三列是残差值
        mean_res = np.mean(res_values)
        max_res = np.max(np.abs(res_values))
        std_res = np.std(res_values)
        
        try:
            import json
            import os
            import random
            residual_file = os.path.join(self.config.results_root or ".", "residual_analyze", "residual_analyze.json")
            if os.path.exists(residual_file):
                with open(residual_file ,"r", encoding="utf-8") as f:
                    experiences = json.load(f)
                
                #提取最后一条信息
                if experiences:
                    last_experience = experiences[-1]
                    last_analysis = last_experience.get("analysis", "")
        except Exception as e:
            # 静默化：简化错误输出
              # print(f"加载残差数据时出错: {str(e)}")
              # print("Error details:")
            traceback.print_exc()


        # 构建分析提示
        res_analyze = f"""
You are a data analysis expert. I will provide a dataset structure for a damped nonlinear oscillator system as follows:
previous conclusions:{last_analysis}
dataset:{residual}
The equation corresponding to the residuals:{sample}

The first two columns are independent variables:
x(position), 
v(velocity).

The third column is the dependent variable a(acceleration).
The forth column contains residuals (calculated as observed value - predicted value from the equation).
Each row represents a set of independent variables (x, v) and their corresponding dependent variable a value, and the residual value.

Task Requirements:

1.Please help me analyze and summarize the influence of the changes in the values of different independent variables on the dependent variable, 
as well as the possible intrinsic relationships among different independent variables.

Your response only needs to answer your analysis results in the form below, and you don't need to show your analysis process.

"""+"""
2.##Output Format##:
STRICTLY deliver results in the following structured format:

Deliver results in the following structured format:

  "output_format": {
    "analysis": {
      "independent_to_dependent_relationships": {
        "x ": [
          "Hint: Here you need to analyze the functional relationship between x and a in different intervals"
        ],
        "v ": [
          "Hint: Here you need to analyze the functional relationship between v and a in different intervals"
        ]
      },
      "inter_relationships_between_independents": {
        "x vs v": [
          "Hint: Here you need to analyze the possible functional relationship between x and v in different intervals. If not, you can leave it blank."
        ]
      }
    }
  }

        """
        
          # 静默化：不显示残差提示词详情
        # print("========这是输入的残差提示词==========\n")
        # print(res_analyze)
        # 调用远程API分析结果
        try:
            client = SHARED_LLM_CLIENT
            if client is None:
                raise ValueError("未注入共享 LLM 客户端，请在 Wrapper 中注入后再运行。")
            resp = client.chat([{"role": "user", "content": res_analyze}])
            analysis_result = resp.get('content', '') or ''
            # 静默化：不显示残差分析结果详情
            # print(f"残差分析结果：{analysis_result}")
            return analysis_result
        except Exception as e:
            # 静默化：简化错误输出
            # print(f"残差分析请求发生错误: {str(e)}")
            return f"分析请求发生错误: {str(e)}"


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    仅提取 equation*(...) 的函数体，丢弃测试代码/顶层语句。
    优先使用 AST 精确定位，失败则回退为基于缩进的保守截断。
    """
    import ast, re

    # 1) AST 路径：找到第一个以 equation 开头的函数，截取其函数体（保持原缩进）
    try:
        tree = ast.parse(sample)
        lines = sample.splitlines()
        target = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('equation'):
                target = node
                break
        if target and target.body:
            body_start = target.body[0].lineno - 1
            body_end = target.end_lineno
            body = "\n".join(lines[body_start:body_end]) + "\n"
            return body
    except SyntaxError:
        pass

    # 2) 回退：从 def equation…: 的下一行开始，收集连续缩进行，遇到顶层非缩进行停止
    lines = sample.splitlines()
    func_def = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith('def ') and re.search(r'\bequation\b', line):
            func_def = i
            break
    if func_def is not None:
        body_lines = []
        indent = '    '
        for line in lines[func_def + 1:]:
            if not line.strip():
                body_lines.append(line)
                continue
            if not line.startswith(indent):
                break
            body_lines.append(line)
        return "\n".join(body_lines) + "\n"

    # 3) 最后回退：返回原样（尽量避免，但保证流程不崩）
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        instruction_prompt = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Complete ONLY the body lines of the 'equation' function below, considering the physical meaning and relationships of inputs.\n"
            "Rules:\n"
            "- Do not output 'def ...' header, imports, tests, prints, or random sampling.\n"
            "- Use variable names 'col0', 'col1', ... that match the function signature.\n"
            "- Return a numpy array with the same length as inputs.\n\n"
        )
        self._batch_inference = batch_inference
        self._instruction_prompt = instruction_prompt
        self._trim = trim

        ####################################
        # 添加会话ID存储
        # self._conversation_ids = {}  # 用于存储每个样本的对话ID
        
        


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # 记录统一结果目录供本地路径引用
        try:
            self._base_dir = config.results_root or "."
        except Exception:
            self._base_dir = "."
        # 绑定共享 client
        self._client = SHARED_LLM_CLIENT

        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:

###############################################
                    # 现在_do_request返回两组响应，只取第一组(方程实现)
                    # 静默化：不显示分支运行信息
                  # print("运行了_draw_samples_local的_batch_inference分支")

                    first_responses = self._do_request(prompt)
                    # 01 版本
                    # first_responses, second_responses = self._do_request(prompt)
                    # 静默化：不显示运行成功信息
                    # print("成功运行first_responses = self._do_request(prompt)")

                    # all_samples = first_responses


                    # 原代码
                    # response = self._do_request(prompt)
                    # for res in response:

#####################################################
                    # 静默化：不显示响应详情
                    # print(first_responses)                   
                    for res in first_responses:
                        all_samples.append(res)

                    # print("-------jjjjjjjjjjjjjjjj")
                else:
                    for _ in range(self._samples_per_prompt):
                        # response = self._do_request(prompt)
                        # all_samples.append(response)
                        # print("________")
                        first_responses, second_responses = self._do_request(prompt)
                        all_samples.append(first_responses)

                # print(all_samples)


                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                # print("处理过的all_samples")
                # print(all_samples)
                

                # 01版本
                # return all_samples, second_responses
            
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        full_prompt = '\n'.join([self._instruction_prompt, prompt])
        client = getattr(self, '_client', None)
        if client is None:
            # 静默化：简化错误输出
                  # print("未注入共享 LLM 客户端，无法进行采样。请在 Wrapper 中通过 set_shared_llm_client 注入。")
            return [""] * self._samples_per_prompt

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    resp = client.chat([{ "role": "user", "content": full_prompt }])
                    content = resp.get('content', '') or ''
                    if self._trim:
                        content = _extract_body(content, config)
                    all_samples.append(content)
                    break
                except Exception as e:
                    # 静默化：简化错误输出
                               # print(f"API请求发生错误: {str(e)}")
                    import time as _t
                    _t.sleep(1)
                    continue
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        # print('ttttttttttttttttttttttttttttttttttttttttt')
        # print(content)
        # print('ttttttttttttttttttttttttttttttttttttttttt')
        # 添加 idea_lib

        # 尝试加载经验数据
        try:
        # if True:


            # 获取当前的 sample_order
            # current_sample_order = self._get_global_sample_nums()
            # print(f"当前 sample_order: {current_sample_order}")
            
            # 计算经验文件中所有类别经验的总数
            current_sample_order = 0

            experience_file = os.path.join(getattr(self, "_base_dir", "."), "equation_experiences", "experiences.json")

            if os.path.exists(experience_file):
                with open(experience_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)
                
                # 统计所有类别的经验总数
                for category in ["None", "Good", "Bad"]:
                    if category in experiences:
                        current_sample_order += len(experiences[category])
                
                # print(f"经验库中共有 {current_sample_order} 条经验")


                
                # 准备存储筛选后的各类经验
                filtered_experiences = {"None": [], "Good": [], "Bad": []}
                
                # 根据当前 sample_order 选择合适的经验
                # for category in ["None", "Good", "Bad"]:
                for category in ["None"]:
                    if category in experiences and experiences[category]:
                        # 筛选符合条件的经验
                        if current_sample_order <= 50:
                            # sample_order < 50 时，不限制经验的 sample_order
                            filtered_category = experiences[category]
                        else:
                            # sample_order > 50 时，只选择 sample_order 在当前值的 0.5~1 倍范围内的经验
                            min_order = current_sample_order * 0.7
                            max_order = current_sample_order
                            filtered_category = [
                                exp for exp in experiences[category] 
                                if "sample_order" in exp and min_order <= exp["sample_order"] <= max_order
                            ]
                        
                        # 随机选择最多3个经验
                        if filtered_category:
                            selected = random.sample(filtered_category, min(3, len(filtered_category)))
                            filtered_experiences[category] = selected
                
                # 合并所有类别的经验
                all_selected_experiences = []
                for category, exps in filtered_experiences.items():
                    for exp in exps:
                        experience_entry = {
                            "type": category,
                            "analysis": exp.get("analysis", ""),
                            "sample_order": exp.get("sample_order", "unknown"),
                        }
                        
                        # 对于 None 类别，添加错误信息（如果有）
                        if category == "None" and "error" in exp:
                            error_msg = exp["error"]
                            # 移除特定错误信息（如果需要）
                            if error_msg == "Execution Error: too many values to unpack (expected 5)":
                                error_msg = ""
                            
                            if error_msg:
                                experience_entry["error"] = error_msg
                        
                        all_selected_experiences.append(experience_entry)
                
                # 如果有经验可用，构建经验提示
                if all_selected_experiences:
                    experience_prompt = "\n\n### The following are ideas summarized based on past experiences in solving such problems. ###\n\n"
                    
                    # 为每个经验分配编号
                    for i, exp in enumerate(all_selected_experiences, 1):
                        experience_prompt += f"idea{i}：\n"
                        # experience_prompt += f"(sample_order: {exp['sample_order']})\n"
                        # 静默化：不显示样本顺序详情
                                 # print("=================================sample_order: ==================================\n", exp['sample_order'])
                        
                        # 限制经验分析文本最多100个字符
                        analysis_text = exp["analysis"] if exp.get("analysis") else ""
                        if len(analysis_text) > 500:
                            analysis_text = analysis_text[:500] + "..."
                        experience_prompt += analysis_text
                        
                        experience_prompt += "\n---\n\n"
                    
                    # 将经验添加到原始内容中
                    content_with_lib = experience_prompt + "\n\n" + content
                    # print(f"已添加 {len(all_selected_experiences)} 条历史经验到提示中")

                    content = content_with_lib

            #有p的几率进入以下代码：
            p = 1.0  # 设置执行概率为50%，你可以根据需要调整这个值
            
            if random.random() < p and os.path.exists(experience_file):
                # 静默化：不显示残差分析使用标记
                  # print("use residual_analyze: True")

                residual_file = os.path.join(getattr(self, "_base_dir", "."), "residual_analyze", "residual_analyze.json")
                if os.path.exists(residual_file):
                    with open(residual_file ,"r", encoding="utf-8") as f:
                        experiences = json.load(f)
                    
                    #提取最后一条信息
                    if experiences:
                        last_experience = experiences[-1]
                        last_analysis = last_experience.get("analysis", "")
                        last_sample_order = last_experience.get("sample_order", "unknown")
                        last_equation = last_experience.get("equation", "")
                        if last_equation is not None:
                        
                            # 构建提示
                            experience_prompt = f"\n\n### The following is the analysis result of the existing data on{problem_name_in_prompt}, which will assist you in answering the question. ###\n\n"
                            # experience_prompt += f"经验{last_sample_order}：\n"
                            # experience_prompt += f"(sample_order: {last_sample_order})\n"
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis
                            # 静默化：不显示样本顺序详情
                                     # print("=================================sample_order: ==================================\n", last_sample_order)
                            # 将经验添加到原始内容中
                            content_with_residual = experience_prompt + "\n\n" + content

                            content = content_with_residual
                        
                            # print(f"残差分析库中共有 {len(experiences)} 条经验")
                        
                        else:
                            # 构建提示
                            experience_prompt = f"\n\n### The following is the analysis result of the existing data on{problem_name_in_prompt}, which will assist you in answering the question. ###\n\n"
                            # experience_prompt += f"经验{last_sample_order}：\n"
                            # experience_prompt += f"(sample_order: {last_sample_order})\n"
                            if isinstance(last_analysis, list):
                                last_analysis = last_analysis[0] if last_analysis else ""
                            # print(f"===========================last_analysis:=====================================\n {last_analysis}")
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis

                            # 将经验添加到原始内容中
                            content_with_residual = experience_prompt + "\n\n" + content

                            content = content_with_residual
                        
                            # print(f"残差分析库中共有 {len(experiences)} 条经验")


        except Exception as e:
            # 静默化：简化错误输出
                # print(f"加载经验数据时出错: {str(e)}")
                # print("Error details:")
            traceback.print_exc()  # 输出详细的错误堆栈信息
        
        # 重复提示以进行批量推理
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1


        head = f"""
        Find the mathematical function skeleton that represents acceleration in{problem_name_in_prompt} with driving force, given data on {independent_name_in_prompt}. 
        """
        content = head +'\n'+ content
        # 静默化：不显示最终输入内容
            # print("========================最终输入给大模型的content========================\n")
            # print(content)

        responses = []
        client = SHARED_LLM_CLIENT
        if client is None:
            # 静默化：简化错误输出
                   # print("未注入共享 LLM 客户端，无法进行请求。请在 Wrapper 中通过 set_shared_llm_client 注入。")
            return [""] * repeat_prompt if self._batch_inference else ""

        for _ in range(repeat_prompt):
            try:
                resp = client.chat([{ "role": "user", "content": content }])
                responses.append(resp.get('content', '') or '')
            except Exception as e:
                # 静默化：简化错误输出
                               # print(f"API请求发生错误: {str(e)}")
                responses.append("")

        return responses if self._batch_inference else responses[0]
