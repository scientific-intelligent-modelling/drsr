"""
集中管理所有 LLM 提示词模板，确保 sampler 与分析提示可配置且与原实现一致。

注意：保持与当前源码中的默认文案完全一致，避免行为差异。
"""

# 任务头中使用的占位参数（用于 _do_request 中的 head 文本格式化）
problem_name_in_prompt = 'a damped nonlinear oscillator system with driving force'
dependent_name_in_prompt = 'acceleration'
independent_name_in_prompt = 'position, and velocity'


# 采样阶段：说明性指令（拼接在代码 prompt 前）
instruction_prompt = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
    "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
)

# 采样后经验对话整体模板（包含上下文与追问占位）
analysis_conversation_template = (
    "Here's our previous conversation:\n\n"
    "user: {prompt}\n\n"
    "assistant: {sample}\n\n"
    "user: {question}\n"
)


# 采样后“基于得分的经验总结”三种追问模板
analysis_question_good = (
    "The optimized function skeleton you just answered scored higher. Please summarize useful experience.\n"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in{problem}, I can ...\"\n"
    "2. Summarize ONLY the key success factors\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_bad = (
    "The optimized function skeleton you just answered scored lower. What lessons can you draw from it?\n"
    "STRICTLY follow these rules: \n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in{problem}, I can ...\"\n"
    "2. Identify ONE crucial improvement point\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_none = (
    "The optimized function skeleton you just answered failed with error: {error}, What lessons can you draw from it?\n"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in{problem}, I need ...\"\n"
    "2. Address the SPECIFIC error: {error}\n"
    "3. Identify ONE crucial improvement point\n"
    "4. You need to make your answer as concise as possible\n"
)

# 经验注入区块标题与条目前缀
ideas_block_title = "\n\n### The following are ideas summarized based on past experiences in solving such problems. ###\n\n"
idea_item_prefix = "idea{index}：\n"

# 残差分析注入区块标题
residual_block_title = ("\n\n### The following is the analysis result of the existing data on{problem}, "
                        "which will assist you in answering the question. ###\n\n")


# 采样阶段：任务头（追加在发送前）
head_template = (
    "Find the mathematical function skeleton that represents {dependent} in{problem} with driving force, "
    "given data on {independent}. \n"
)


# 残差分析提示模板（包含固定格式与输出要求）
residual_analysis_prompt = (
    "You are a data analysis expert. I will provide a dataset structure for a damped nonlinear oscillator system as follows:\n"
    "previous conclusions:{last_analysis}\n"
    "dataset:{residual}\n"
    "The equation corresponding to the residuals:{sample}\n\n"
    "The first two columns are independent variables:\n"
    "x(position), \n"
    "v(velocity).\n\n"
    "The third column is the dependent variable a(acceleration).\n"
    "The forth column contains residuals (calculated as observed value - predicted value from the equation).\n"
    "Each row represents a set of independent variables (x, v) and their corresponding dependent variable a value, and the residual value.\n\n"
    "Task Requirements:\n\n"
    "1.Please help me analyze and summarize the influence of the changes in the values of different independent variables on the dependent variable, \n"
    "as well as the possible intrinsic relationships among different independent variables.\n\n"
    "Your response only needs to answer your analysis results in the form below, and you don't need to show your analysis process.\n\n"
    "2.##Output Format##:\n"
    "STRICTLY deliver results in the following structured format:\n\n"
    "Deliver results in the following structured format:\n\n"
    "  \"output_format\": {\n"
    "    \"analysis\": {\n"
    "      \"independent_to_dependent_relationships\": {\n"
    "        \"x \": [\n"
    "          \"Hint: Here you need to analyze the functional relationship between x and a in different intervals\"\n"
    "        ],\n"
    "        \"v \": [\n"
    "          \"Hint: Here you need to analyze the functional relationship between v and a in different intervals\"\n"
    "        ]\n"
    "      },\n"
    "      \"inter_relationships_between_independents\": {\n"
    "        \"x vs v\": [\n"
    "          \"Hint: Here you need to analyze the possible functional relationship between x and v in different intervals. If not, you can leave it blank.\"\n"
    "        ]\n"
    "      }\n"
    "    }\n"
    "  }\n"
)


# ==========================
# 动态渲染：无装饰器版本的上下文类
# ==========================

DEFAULT_BACKGROUND = "The physical properties of this equation are unknown and need to be analyzed based on experience."

def _ensure_feature_names(n, names):
    """确保有 n 个自变量名；缺省则按 x1..xN 生成。"""
    if names is None:
        return [f"x{i+1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
    return names

def _ind_phrase(names):
    """生成 “x1, x2, and x3” 风格短语。"""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def _pairwise(names):
    """两两组合。"""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


class PromptContext:
    """提示词渲染上下文（无装饰器版本）。

    用法：
        ctx = PromptContext(n_features=X.shape[1], feature_names=None, dependent_name=None,
                            problem_name=None, background=None)
        head = ctx.render_head()
        instruction = ctx.render_instruction()
        q = ctx.render_analysis_question('Good')
        residual_prompt = ctx.render_residual_analysis_prompt(last_analysis, residual, sample)
    """

    def __init__(self, n_features, feature_names=None, dependent_name=None, problem_name=None, background=None):
        self.n_features = n_features
        self.feature_names = feature_names
        self.dependent_name = dependent_name
        self.problem_name = problem_name
        self.background = background

    # 规范化后的属性
    @property
    def features(self):
        return _ensure_feature_names(self.n_features, self.feature_names)

    @property
    def dependent(self):
        return self.dependent_name or "y"

    @property
    def problem(self):
        return self.problem_name or problem_name_in_prompt

    @property
    def background_text(self):
        return (self.background or DEFAULT_BACKGROUND).strip()

    # 渲染方法
    def render_instruction(self):
        vars_line = f"Variables: {', '.join(self.features)} -> {self.dependent}\n"
        bg_line = f"Background: {self.background_text}\n"
        return instruction_prompt + vars_line + bg_line

    def render_head(self):
        return head_template.format(
            dependent=self.dependent,
            problem=self.problem,
            independent=_ind_phrase(self.features),
        )

    def render_analysis_question(self, quality, error=None):
        if quality == "Good":
            return analysis_question_good.format(dependent=self.dependent, problem=self.problem)
        if quality == "Bad":
            return analysis_question_bad.format(dependent=self.dependent, problem=self.problem)
        if quality == "None":
            return analysis_question_none.format(
                dependent=self.dependent, problem=self.problem, error=str(error or "")
            )
        raise ValueError(f"unknown quality: {quality}")

    def render_residual_block_title(self):
        return residual_block_title.format(problem=self.problem)

    def render_residual_analysis_prompt(self, last_analysis, residual, sample):
        inds = self.features
        dep = self.dependent

        role_lines = [
            "The independent variables are:",
            *inds,
            "",
            f"The dependent variable is {dep}.",
            "The forth column contains residuals (observed - predicted).",
        ]
        role_text = "\n".join(role_lines)

        ind_to_dep = "\n".join([
            f'        "{name} ": [\n'
            f'          "Hint: analyze the functional relationship between {name} and {dep} in different intervals"\n'
            f"        ],"
            for name in inds
        ])

        pairs = _pairwise(inds)
        inter_lines = "\n".join([
            f'        "{a} vs {b}": [\n'
            f'          "Hint: analyze possible functional relationship between {a} and {b} in different intervals. If not, leave blank."\n'
            f"        ]"
            for a, b in pairs
        ])
        if not inter_lines:
            inter_lines = '        "": []'

        dynamic_part = (
            f"{role_text}\n\n"
            "Task Requirements:\n\n"
            "1. Analyze and summarize how changes of each independent variable influence the dependent variable, "
            "and the possible intrinsic relationships among independent variables.\n\n"
            "Your response should follow the structure below; no need to show the reasoning process.\n\n"
            '2.##Output Format##:\n'
            'STRICTLY deliver results in the following structured format:\n\n'
            '  "output_format": {\n'
            '    "analysis": {\n'
            '      "independent_to_dependent_relationships": {\n'
            f"{ind_to_dep}\n"
            '      },\n'
            '      "inter_relationships_between_independents": {\n'
            f"{inter_lines}\n"
            '      }\n'
            '    }\n'
            '  }\n'
        )

        return (
            "You are a data analysis expert.\n"
            f"Background: {self.background_text}\n"
            f"previous conclusions:{last_analysis}\n"
            f"dataset:{residual}\n"
            f"The equation corresponding to the residuals:{sample}\n\n"
            f"{dynamic_part}"
        )
