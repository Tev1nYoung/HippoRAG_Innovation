# AGENTS.md

本文件为 Codex CLI（OpenAI Codex）在本仓库进行开发与实验时提供提示与约定。

## 项目概述

HippoRAG 2 是一个受神经生物学启发的图增强RAG框架，通过OpenIE构建知识图谱来增强LLM的长期记忆和多跳推理能力。

**核心论文:**
- HippoRAG 1 (NeurIPS '24): https://arxiv.org/abs/2405.14831
- HippoRAG 2 (ICML '25): https://arxiv.org/abs/2502.14802

## 环境设置

```bash
# 创建环境
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag

# 或从源码安装
pip install -e .

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your api key>  # 支持第三方兼容OpenAI格式的API
export TOKENIZERS_PARALLELISM=false
```

## 第三方API配置（重要）

**本项目使用多个第三方API服务商，配置信息存储在 `KEY.txt` 文件中。**

### API配置说明
- **统一使用模型**: Llama-3.3-70B 用于三元组构造（OpenIE）
- **多API切换**: 因rpm限制和额度不同，会在多个API服务商之间切换
- **模型名称差异**: 不同服务商对同一模型的命名不同，需要严格匹配

### 基本使用格式
```bash
export OPENAI_API_KEY="<your_api_key>"
python main.py \
    --dataset <dataset_name> \
    --llm_name <model_name_at_provider> \
    --llm_base_url "<api_url>" \
    --embedding_name <embedding_model>
```

### 常用数据集
```
sample, musique, 2wikimultihopqa, hotpotqa, nq_rear, popqa
```

### 常用Embedding模型
```
facebook/contriever
colbert-ir/colbertv2.0
```

### 数据复用机制（重要！节省API成本）

当切换API服务商但使用相同的底层模型时，可以复用已生成的OpenIE结果和检索器数据，避免重复消耗API额度。

**操作步骤：**
```bash
# 场景：从服务商A（模型名llama3.3-70b）切换到服务商B（模型名meta-llama/llama-3.3-70b-instruct:free）
# 因为底层模型相同，可以复用数据

# 1. 重命名OpenIE结果文件
cd reproduce/dataset/openie_results/
mv openie_results_ner_llama3.3-70b.json \
   openie_results_ner_meta-llama_llama-3.3-70b-instruct:free.json

# 2. 重命名输出目录
cd outputs/<dataset>/
mv llama3.3-70b_facebook_contriever \
   meta-llama_llama-3.3-70b-instruct:free_facebook_contriever
```

**注意事项：**
- 文件名中的斜杠 `/` 会被替换为下划线 `_`
- 必须确保底层模型能力相同才能复用
- 复用可以显著节省API调用成本和时间

### 安全提示
- `KEY.txt` 包含敏感API密钥，**切勿提交到Git仓库**
- 建议将 `KEY.txt` 添加到 `.gitignore`
- 定期轮换API密钥

## 测试命令

### 第三方API测试（推荐用于API用户）
```bash
# 适用于兼容OpenAI格式的第三方API（如DeepSeek、Moonshot、智谱等）
export OPENAI_API_KEY=<your third-party api key>

# 示例1: 使用DeepSeek
python tests_third_party_api.py \
    --llm_base_url https://api.deepseek.com/v1 \
    --llm_name deepseek-chat \
    --embedding_base_url https://api.deepseek.com/v1 \
    --embedding_name text-embedding-3-small

# 示例2: 使用其他兼容OpenAI格式的API
python tests_third_party_api.py \
    --llm_base_url <your_api_url> \
    --llm_name <model_name> \
    --embedding_base_url <your_embedding_url> \
    --embedding_name <embedding_model_name>
```

### OpenAI官方API测试
```bash
export OPENAI_API_KEY=<your openai api key>
python examples/tests_openai.py
```

### 本地vLLM测试（需要GPU资源）
```bash
# 终端1: 启动vLLM服务器
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve meta-llama/Llama-3.1-8B-Instruct --max_model_len 4096 --gpu-memory-utilization 0.95 --port 6578

# 终端2: 运行测试
CUDA_VISIBLE_DEVICES=1 python examples/tests_local.py
```

## 运行实验

### 使用OpenAI模型
```bash
dataset=sample  # 或 musique, hotpotqa, 2wikimultihopqa 等
python main.py --dataset $dataset \
    --llm_base_url https://api.openai.com/v1 \
    --llm_name gpt-4o-mini \
    --embedding_name nvidia/NV-Embed-v2
```

### 使用本地vLLM (在线模式)
```bash
# 终端1: 启动vLLM服务器
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95

# 终端2: 运行实验
export CUDA_VISIBLE_DEVICES=2,3
python main.py --dataset $dataset \
    --llm_base_url http://localhost:8000/v1 \
    --llm_name meta-llama/Llama-3.3-70B-Instruct \
    --embedding_name nvidia/NV-Embed-v2
```

### 使用vLLM离线批处理模式 (3倍加速)
```bash
# 步骤1: 离线OpenIE (使用所有GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python main.py --dataset $dataset \
    --llm_name meta-llama/Llama-3.3-70B-Instruct \
    --openie_mode offline \
    --skip_graph

# 步骤2: 启动vLLM服务器并运行完整流程 (参考上面的在线模式)
```

### 重新运行实验 (清除缓存)
```bash
# 清除OpenIE结果和图数据
rm reproduce/dataset/openie_results/openie_${dataset}_results_ner_${llm_name}.json
rm -rf outputs/${dataset}/${llm_name}_${embedding_name}
```

## 核心架构

### 三阶段工作流程

**1. 索引阶段 (Indexing)**
```
文档 → 分块 → OpenIE抽取 → 构建知识图谱 → 存储embeddings
```
- OpenIE使用LLM提取实体和三元组关系
- 构建图: 实体节点 + 段落节点 + 三种边 (Fact/Passage/Synonymy)
- 存储三种embeddings: chunk, entity, fact

**2. 检索阶段 (Retrieval)**
```
查询 → Fact检索 → Recognition Memory过滤 → PPR图搜索 → 返回文档
```
- Fact检索: 查询与fact embeddings相似度匹配
- Recognition Memory: LLM重排序和过滤facts
- PPR (Personalized PageRank): 在知识图谱上运行PPR算法

**3. 问答阶段 (QA)**
```
检索文档 → 构建Prompt → LLM生成答案 → 解析输出
```

### 关键类和文件

**核心API类:**
- `src/hipporag/HippoRAG.py` (1739行): 主要API入口
  - `index(docs)`: 索引文档，构建知识图谱
  - `retrieve(queries, num_to_retrieve)`: HippoRAG检索
  - `rag_qa(queries, gold_docs, gold_answers)`: 检索增强问答
  - `delete(doc_indices)`: 删除文档
  - `run_ppr()`: 运行Personalized PageRank算法
  - `add_fact_edges()`, `add_passage_edges()`, `add_synonymy_edges()`: 图构建

- `src/hipporag/StandardRAG.py` (518行): 标准DPR基线实现

**存储和Embedding:**
- `src/hipporag/embedding_store.py`: EmbeddingStore类，管理Parquet格式的embeddings
- 存储结构: `outputs/<dataset>/<llm>_<embedding>/`
  - `chunk_embeddings/vdb_chunk.parquet`
  - `entity_embeddings/vdb_entity.parquet`
  - `fact_embeddings/vdb_fact.parquet`
  - `graph.pickle`: 知识图谱 (igraph格式)

**配置系统:**
- `src/hipporag/utils/config_utils.py`: BaseConfig类
  - 关键配置: llm_name, embedding_model_name, openie_mode, retrieval_top_k, damping, passage_node_weight

**模块化组件:**
- `src/hipporag/llm/`: LLM推理 (OpenAI, vLLM, Transformers, Bedrock)
- `src/hipporag/embedding_model/`: Embedding模型 (NV-Embed-v2, GritLM, Contriever, OpenAI)
- `src/hipporag/information_extraction/`: OpenIE实现 (在线/离线模式)
- `src/hipporag/evaluation/`: 评估指标 (Recall@k, ExactMatch, F1)
- `src/hipporag/prompts/`: Prompt模板管理系统
- `src/hipporag/rerank.py`: DSPyFilter重排序

### 数据格式

**语料库** (`reproduce/dataset/<dataset>_corpus.json`):
```json
[
  {"title": "标题", "text": "内容", "idx": 0}
]
```

**查询集** (`reproduce/dataset/<dataset>.json`):
```json
[
  {
    "question": "问题",
    "answer": ["答案"],
    "paragraphs": [
      {"title": "...", "text": "...", "is_supporting": true}
    ]
  }
]
```

## 设计模式和扩展点

**工厂模式**: LLM和Embedding模型通过`_get_llm_class()`和`_get_embedding_model_class()`动态加载

**策略模式**: 不同检索策略 (HippoRAG vs StandardRAG)

**模板方法**: BaseMetric评估框架，BaseLLM和BaseEmbeddingModel基类

**关键扩展点:**
1. 添加新的LLM后端: 继承`BaseLLM`并在`llm/__init__.py`注册
2. 添加新的Embedding模型: 继承`BaseEmbeddingModel`并在`embedding_model/__init__.py`注册
3. 修改图构建策略: 修改`HippoRAG.py`中的`add_*_edges()`方法
4. 自定义检索算法: 修改`run_ppr()`或创建新的检索方法
5. 添加新的评估指标: 继承`BaseMetric`

## 调试技巧

**使用小数据集调试:**
```bash
python main.py --dataset sample --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

**调试vLLM离线模式:**
- 在`src/hipporag/llm/vllm_offline.py`中设置`tensor_parallel_size=1`

**日志级别:**
```bash
export LOG_LEVEL=DEBUG
```

**常见问题:**
- OOM错误: 调整`--gpu-memory-utilization`或`--max_model_len`
- vLLM多进程问题: 确保设置`export VLLM_WORKER_MULTIPROC_METHOD=spawn`
- Tokenizer警告: 设置`export TOKENIZERS_PARALLELISM=false`

## 性能优化

**硬件配置建议 (RTX 4070 8GB):**
- 使用较小模型: Llama-3.1-8B-Instruct
- vLLM离线模式: 3倍以上加速
- 调整batch_size: `embedding_batch_size=16` (默认)
- 单GPU部署: `tensor-parallel-size=1`

**缓存机制:**
- LLM调用自动缓存到SQLite (通过litellm)
- OpenIE结果缓存到`reproduce/dataset/openie_results/`
- 设置`force_openie_from_scratch=false`重用OpenIE结果

## 重要注意事项

1. **OpenIE是最耗时的步骤**: 使用vLLM离线模式或缓存OpenIE结果
2. **图构建参数影响性能**: `synonymy_edge_topk`, `synonymy_edge_sim_threshold`, `damping`, `passage_node_weight`
3. **增量更新支持**: 可以使用`delete()`删除文档后重新索引
4. **多后端支持**: 代码设计支持OpenAI、vLLM、Transformers、Azure、Bedrock
5. **模块化设计**: 各组件高度解耦，易于替换和扩展
