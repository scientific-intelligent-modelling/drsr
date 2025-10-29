import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取JSON文件
json_path = '/data/home/zdhs0036/DrSR/equation_experiences/experiences.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# 提取Good数据并检查有效性
good_data = []
for item in data.get('Good', []):
    if 'score' in item and 'sample_order' in item:
        good_data.append((item['sample_order'], -item['score']))  # 分数取负号反转

if not good_data:
    print("警告：未找到有效的Good样本数据！")
    exit()

# 按sample_order排序
good_data.sort(key=lambda x: x[0])
sample_orders = [item[0] for item in good_data]
scores = [item[1] for item in good_data]



# 计算历史最优分数（最大值）
best_scores = []
current_best = float('inf')  # 初始化为负无穷
for score in scores:
    current_best = min(score, current_best)
    best_scores.append(current_best)

# 将分数转换为对数值（处理可能的零或负值）
log_best_scores = []
for score in best_scores:

    if score <= 0:
        log_score = float('nan')  # 对于零或负值，设为NaN
    else:
        log_score = np.log10(score)
    log_best_scores.append(log_score)

# 绘制折线图（直接使用对数值）
plt.figure(figsize=(12, 8))
plt.plot(sample_orders, log_best_scores, linestyle='-', color='blue', linewidth=2,
         label=f'Log10(Best Score) (Max={best_scores[-1]:.2e})')  # 科学计数法显示最大原始分数

# 设置坐标轴和标题
plt.title('Log10 of Best Score So Far vs Sample Order', fontsize=16)
plt.xlabel('Sample Order', fontsize=14)
plt.ylabel('Log10(Score) (Higher is Better)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 动态设置横轴刻度
min_order, max_order = min(sample_orders), max(sample_orders)
step = max(1, (max_order - min_order) // 10)
plt.xticks(np.arange(min_order, max_order + step, step))

# 保存图像并输出结果
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'best_scores_plot_log.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# 输出统计信息
print(f"折线图已保存至 {output_path}")
print(f"统计信息:\n"
      f"- 总样本数: {len(best_scores)}\n"
      f"- 最终最优分数: {best_scores[-1]:.2e}\n"
      f"- 最优分数首次出现位置: sample_order={sample_orders[best_scores.index(best_scores[-1])]}")