from argparse import ArgumentParser
from flask import Flask, request, jsonify
from flask_cors import CORS
from vllm import LLM, SamplingParams

app = Flask(__name__)
CORS(app)

# 参数解析
parser = ArgumentParser()
parser.add_argument('--gpu_ids', nargs='+', default=['0','1','2','3'])
parser.add_argument('--model_path', type=str, default='/data/home/zdhs0036/huggingface_models/Mixtral-8x7B-Instruct-v0.1')
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--eos_token_id', type=int, default=32021)
parser.add_argument('--pad_token_id', type=int, default=32021)
parser.add_argument('--num_return_sequences', type=int, default=1)
parser.add_argument('--max_repeat_prompt', type=int, default=10)
args = parser.parse_args()

# 初始化vLLM
llm = LLM(
    model=args.model_path,
    tensor_parallel_size=len(args.gpu_ids),
    trust_remote_code=True,
    gpu_memory_utilization=0.6
)

@app.route('/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']
    chat_prompt = [{'role': 'user', 'content': prompt}]
    
    # 生成对话模板
    formatted_prompt = llm.engine.tokenizer.apply_chat_template(
        chat_prompt,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # 处理生成参数
    params = content.get('params', {})
    sampling_params = SamplingParams(
        n=content.get('repeat_prompt', 1),
        temperature=params.get('temperature', args.temperature),
        top_p=params.get('top_p', args.top_p),
        top_k=params.get('top_k', args.top_k),
        max_tokens=params.get('max_new_tokens', args.max_new_tokens),
        stop_token_ids=[params.get('eos_token_id', args.eos_token_id)],
        skip_special_tokens=True
    )
    
    # 生成响应
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    # 处理输出结果
    responses = []
    for output in outputs:
        for choice in output.outputs:
            responses.append(choice.text.strip())
    
    return jsonify({'content': responses})

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)