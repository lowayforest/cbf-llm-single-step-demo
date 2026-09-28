# CBF-LLM：小模型单步机制演示

这个目录用小型生成模型 `distilbert/distilgpt2` 与论文同款情感分类器 `cardiffnlp/twitter-roberta-base-sentiment-latest`，实现单步 CBF 过滤。生成模型不同于论文的 Llama 3 8B，因此这是**机制级复现**，不能复刻论文 Table I、II 的数值。

## 已完成实验：环境与结论

本次运行环境是 Apple M5、16 GB 内存、macOS 26.5.1（arm64）；虚拟环境 Python 3.12.14、PyTorch 2.14.0、Transformers 5.17.0、Matplotlib 3.11.2。程序本次选择了 `cpu`，没有使用 MPS/GPU。起始提示词为 `I feel hopeful about the future, but`，随机种子为 7；三种设置各生成 20 个 token。

| 设置 | 最低 $h$ | 最终 $h$ | 被拒绝候选 | 用时 |
|---|---:|---:|---:|---:|
| 无干预 | −0.590 | −0.357 | 0 | 1.0 秒 |
| CBF $\gamma=0$ | 0.649 | 0.688 | 2 | 3.6 秒 |
| CBF $\gamma=0.4$ | 0.027 | 0.027 | 109 | 3.5 秒 |

无干预轨迹在第 7 个生成 token 后第一次跌破 $h=0$，两条受控轨迹始终非负。程序逐步核对 $h_{k+1}\ge\gamma h_k$，因此这组运行展示了过滤机制的定性效果。$\gamma=0.4$ 与 $\gamma=0$ 走上不同文本轨迹，不能直接比较最终 $h$ 来断言哪个“更安全”。本实验只有一组有意挑选的提示词和种子，不能据此估计总体成功率、证明普遍安全，或声称复刻论文表格数值。

图和逐 token 数据见 [实验结果.md](results/实验结果.md)、[h_trajectory.png](results/h_trajectory.png)、[trajectory.csv](results/trajectory.csv) 与 [results.json](results/results.json)。

## 运行

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在本仓库根目录执行（以下命令适用于 macOS/Linux）：

```bash
UV_CACHE_DIR="$PWD/.repro_cache" uv venv .venv --python 3.12
UV_CACHE_DIR="$PWD/.repro_cache" uv pip install --python .venv/bin/python \
  -r requirements.txt

HF_HOME="$PWD/.repro_cache/hf" .venv/bin/hf download distilbert/distilgpt2 \
  --include '*.json' --include '*.safetensors' --include '*.txt' \
  --local-dir models/distilgpt2

HF_HOME="$PWD/.repro_cache/hf" .venv/bin/hf download \
  cardiffnlp/twitter-roberta-base-sentiment-latest \
  --include '*.json' --include '*.safetensors' --include '*.txt' \
  --local-dir models/roberta-sentiment
HF_HOME="$PWD/.repro_cache/hf" .venv/bin/hf download \
  cardiffnlp/twitter-roberta-base-sentiment-latest pytorch_model.bin \
  --local-dir models/roberta-sentiment

.venv/bin/python run_demo.py
```

运行会在 `results/` 生成或更新：

- `实验结果.md`：课堂上可阅读的结果总结；
- `h_trajectory.png`：无干预与 CBF 的分数曲线；
- `trajectory.csv`：每一步的分数；
- `results.json`：逐步选择、拒绝数量和运行参数。

## 演示的核心规则

对每个候选 token 计算接上它后的情感分数：

$$
h(x_k\oplus t)=p_{\text{positive}}-\max(p_{\text{negative}},p_{\text{neutral}}).
$$

仅允许满足 $h(x_k\oplus t)\ge\gamma h(x_k)$ 的 token。合格候选按原模型概率比例重新抽样。程序先运行无干预，再运行 $\gamma=0$ 和 $\gamma=0.4$，使用同一提示词和随机种子。起始文本必须满足 $h(x_0)\ge0$。

每步最多检查原模型概率最高的 50 个候选，并尝试收集 5 个合格候选。这个搜索上限是为了快速演示，和论文完整扫描算法不同。如果 50 个候选里一个都不合格，程序记录原因并停止，不选择违规 token；如果有合格候选但不足 5 个，则从已有合格候选中抽样，并在结果中记录未收满的步数。

## 快速调整

```bash
.venv/bin/python run_demo.py \
  --seed 11 --max-new-tokens 25 --gammas 0 0.4 \
  --prompt 'I really appreciate your help, but'
```

如果某个提示词的起始分数为负，程序会拒绝运行；换一个积极开头即可。默认设置会从六个英文开头中选择 **正分数最低** 的一个，并在结果文件里记录所选提示词。这个选择用于展示机制，属于有意挑选样例，不代表随机测试集。
