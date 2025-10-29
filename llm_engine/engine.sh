cd /data/home/zdhs0037/DrSR/llm_engine

conda activate llmsr

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5000_gpu0.py >5000.out &

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5001_gpu1.py >5001.out &

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5002_gpu2.py >5002.out &

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5000_gpu_all.py >5000_all.out &

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5001_gpu1_llama.py >5001.out &

nohup python3 /data/home/zdhs0037/DrSR/llm_engine/engine_5003_gpu2_llama.py >5003.out &

cd /data/home/zdhs0037/DrSR
nohup python3 main_5000.py >backgrow_mix.out &

cd /data/home/zdhs0037/DrSR_5001
nohup python3 main_5001.py >backgrow_llama.out &

cd /data/home/zdhs0037/DrSR_5002
nohup python3 main.py >mix_p=1_drsr_question2.out &

cd /data/home/zdhs0037/DrSR_5003
nohup python3 main_5003.py >strss_llama.out &
