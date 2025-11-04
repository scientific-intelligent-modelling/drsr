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

""" Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing

from drsr_420 import code_manipulation
from drsr_420 import buffer
from drsr_420 import evaluator_accelerate
from drsr_420 import evaluate_on_problems

class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _is_invalid_equation(sample: str) -> bool:
    """快速门禁：过滤明显不合法/危险样本，避免进入沙箱执行。

    注意：允许"仅函数体"的样本（不强制包含 def equation），因此不检查 'equation' 关键词。
    仅做黑名单过滤与空白检查。
    """
    if not sample or not sample.strip():
        return True
    blacklist = [
        'import ', 'print(', 'open(', 'os.', 'sys.', '__import__', 'subprocess', 'eval(', 'exec(', 'if __name__', 'while True:'
    ]
    lowered = sample.strip()
    for token in blacklist:
        if token in lowered:
            return True
    return False


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs

    # ) -> tuple[Any, bool]:

    # 02 版本 输出报错信息
    ) -> tuple[Any, bool, str]:
        
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.
        
        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility). 
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

#################################### 02版本
    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, Any]:
        # 原版
        # inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.
        
        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """

        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue() 
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        # if the process is not finished in time, terminate
        if process.is_alive():
            process.terminate()
            process.join()
            results = None, None, False, 'timeout01', None
        else:
            results = self._get_results(result_queue)
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)
        # 解析五元组 (score, residual_sampled, runs_ok, remark, opt_params)
        if results and len(results) == 5:
            grade, res, runs_ok, remark, opt_params = results
            results = (grade, runs_ok, remark, opt_params)
        else:
            res = None
            grade, runs_ok, remark, opt_params = None, False, 'bad_result', None
            results = (grade, runs_ok, remark, opt_params)
            # print('*********************')
            # print(self._print_evaluation_details(program, results, **kwargs))
        # print("result:------------")
        # print(results)

        return results, res


    def _get_results(self, queue):
        # 尝试一次获取队列结果
        for _ in range(1):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, None, False, 'timeout02', None


    def _print_evaluation_details(self, program, results, **kwargs):
        pass
        # 静默化：不显示评估程序详情
        # print('================= Evaluated Program =================')
        # function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        # print(f'{str(function).strip()}\n-----------------------------------------------------')
        # print(f'Score: {results}\n=====================================================\n\n')

    def _clean_program_text(self, program: str, function_name: str) -> str:
        """
        清理程序文本，移除 LLM 可能在函数体后添加的测试代码。
        只保留到目标函数的 return 语句为止的内容。
        """
        try:
            # 解析 AST
            tree = ast.parse(program)
            
            # 找到目标函数节点
            target_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_func = node
                    break
            
            if target_func is None:
                # 找不到目标函数，返回原始程序
                return program
            
            # 获取函数结束行号
            func_end_line = target_func.end_lineno
            
            # 按行分割程序
            lines = program.split('\n')
            
            # 只保留到函数结束行的内容，再加上其他非函数定义的全局代码
            # 简单策略：保留从开头到函数结束行，丢弃之后的内容
            # 但要保留前面的 import 和全局变量定义
            
            # 找到函数开始行
            func_start_line = target_func.lineno
            
            # 保留：1) 函数之前的所有内容  2) 函数本身  3) 丢弃函数之后的测试代码
            cleaned_lines = lines[:func_end_line]
            
            return '\n'.join(cleaned_lines)
            
        except Exception:
            # 如果清理失败，返回原始程序
            return program



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            # [PATCH] 清理方程中的测试代码，避免 LLM 生成的额外代码导致编译失败
            program = self._clean_program_text(program, function_to_evolve)
            
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            evolved_function = all_globals_namespace[function_to_evolve]
            score, full_res, opt_params = evaluate_on_problems.evaluate(dataset, evolved_function)
            if full_res is not None and hasattr(full_res, "shape") and len(full_res) > 0:
                import numpy as np
                # 确定采样数量，最多取20个点或全部（如果数据量小于20）
                sample_size = min(100, len(full_res))
                # 随机选择索引，不排序，保持随机性
                indices = np.random.choice(len(full_res), sample_size, replace=False)
                # 对残差进行采样
                res = full_res[indices]
                # print(f"残差随机采样（{sample_size}个点）:", res)
            if not isinstance(score, (int, float)):
                result_queue.put((None, None, False, 'no output', None))
                return
            
            # 返回 (score, residual_sampled, runs_ok, remark, opt_params)
            result_queue.put((score, res, True, 'yes', opt_params))
            
        # if raise any exception, execution is failed
        except Exception as e:
            # print(f"Execution Error: {e}")
            # result_queue.put((None, False))

            error_msg = f"Execution Error: {e}"
            # 静默化：不显示错误调试信息
                       # print('eeeeeeerrrrrrrrrroooooorrrrrrrr')
            print(error_msg)
            result_queue.put((None, None, False, error_msg, None))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    # ) -> None:
    
    # ) -> float:
    ) -> tuple[float, str, Any]:
        
        # tuple[Any, bool, str]
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        # 先做一次门禁过滤，避免执行无关/危险代码
        if _is_invalid_equation(sample):
            return None, 'invalid_equation', None, None

        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}

        time_reset = time.time()

        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(self._inputs) # x1 x2
        '''
        {'data': {'inputs': array([[-0.25197899, -0.17306601],
       [-0.25232508, -0.17300887],
       [-0.25267104, -0.17295167],
       ...,
       [-0.41992701,  0.11309208],
       [-0.41970063,  0.11328232],
       [-0.41947387,  0.11347256]]), 'outputs': array([0.0285521 , 0.02858525, 0.02861839, ..., 0.09512003, 0.09511695,
       0.09511373])}}
        '''
        
        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(bbbbb)
        opt_params = None
        for current_input in self._inputs:
            
            # test_output, runs_ok = self._sandbox.run(


            # 02 版本 收集错误信息
            # test_output, runs_ok, error_msg = self._sandbox.run(
            results, res = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            test_output, runs_ok, error_msg, opt_params = results
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if not isinstance(test_output, (int, float)):
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset
        ###################
        # print("error_msg=========")
        # print(error_msg)
        # print(test_output)      # score: -0.0004185108785400066 为针对初始化方程框架的评分
        # print('我从analyse中拿到了res', res) 
        # print(bbb)


        # 果代码运行成功并得到有效评分，分数会被保存到经验缓冲区(ExperienceBuffer)：
        '''
        这里的_database就是从sampler.py传入的buffer.ExperienceBuffer实例。它将：

        将函数与其评分一起保存
        将函数分配到适当的"岛屿"(island)中
        根据功能相似性将函数组织到集群(clusters)中
        '''
        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
        
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)
        
        
        return test_output, error_msg, res, opt_params
