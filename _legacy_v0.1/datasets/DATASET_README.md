
## 一、对外接口


  

### 1. 构建完整 CMRC 数据与向量库

  

**函数**  

  

```python

from qwen_rag_eval.vector.vector_builder import build_cmrc_dataset_and_vector_store

  

def build_cmrc_dataset_and_vector_store(

    config_path: str = "config/application.yaml",

) -> None: ...

```

  

**功能说明**  

  

1. 按配置从原始 CMRC 数据开始，依次完成  

   raw → samples → chunks → Chroma 向量库。  

2. 若中间产物已存在，则在保证一致性的前提下复用，不重复构建。

  

**典型用法**  

  

```python

from qwen_rag_eval.vector.vector_builder import build_cmrc_dataset_and_vector_store

  

if __name__ == "__main__":

    build_cmrc_dataset_and_vector_store("config/application.yaml")

```

  

### 2. 基于任意 samples 一键构建向量库

  

**函数**  

  

```python

from qwen_rag_eval.vector.vector_builder import build_vector_store_from_samples

  

def build_vector_store_from_samples(

    samples_path: str | Path,

    config_path: str = "config/application.yaml",

    collection_name: str | None = None,

    overwrite: bool = False,

) -> None: ...

```

  

**功能说明**  

  

1. 假定 `samples_path` 指向的 JSON 文件已经满足本 README 约定的 samples 数据结构。  

2. 根据该文件自动生成 chunks，并写入向量库。  

3. 若指定集合已存在且 `overwrite=False`，则直接返回；`overwrite=True` 时会先删除再重建。

  

**典型用法**  

  

```python

from qwen_rag_eval.vector.vector_builder import build_vector_store_from_samples

  

if __name__ == "__main__":

    build_vector_store_from_samples(

        "datasets/processed/my_samples.json",

        config_path="config/application.yaml",

        # collection_name="my_collection",  # 可选

        overwrite=False,

    )

```

  

### 3. 只负责“确保 CMRC 向量库存在”

  

**函数**  

  

```python

from qwen_rag_eval.vector.vector_builder import ensure_cmrc_vector_store

  

def ensure_cmrc_vector_store(

    config: YamlConfigReader | str = "config/application.yaml",

) -> None: ...

```

  

**功能说明**  

  

1. 根据配置检查目标 Chroma collection 是否存在。  

2. 不存在时：若 samples 或 chunks 缺失，会自动调用相关构建函数，并最终写入向量库。  

3. 已存在时：直接返回，不进行任何修改。

  

**典型用法**  

  

```python

from utils import YamlConfigReader

from qwen_rag_eval.vector.vector_builder import ensure_cmrc_vector_store

  

if __name__ == "__main__":

    config = YamlConfigReader("config/application.yaml")

    ensure_cmrc_vector_store(config)

```

  

### 4. 从原始数据构建 CMRC 评估样本（samples）

  

**函数**  

  

```python

from qwen_rag_eval.dataset_tools.sampling import build_eval_samples

  

def build_eval_samples(

    config: YamlConfigReader | str = "config/application.yaml",

) -> Path: ...

```

  

**功能说明**  

  

1. 从 `dataset.raw_path` 读取原始 CMRC 数据。  

2. 抽取并转换为标准化的评估样本结构，写入 `dataset.samples_path`。  

3. 如目标文件已存在，则直接返回该路径。

  

**典型用法**  

  

```python

from qwen_rag_eval.dataset_tools.sampling import build_eval_samples

  

if __name__ == "__main__":

    samples_file = build_eval_samples("config/application.yaml")

    print("samples 路径:", samples_file)

```

  

### 5. 从 samples 构建 chunks 文件

  

**函数**  

  

```python

from qwen_rag_eval.dataset_tools.chunking import make_chunks_from_samples

  

def make_chunks_from_samples(

    samples_path: str | Path,

    chunks_path: str | Path,

    chunk_size: int = 300,

    chunk_overlap: int = 30,

) -> Path: ...

```

  

**功能说明**  

  

1. 针对每条样本的 `ground_truth_context` 文本进行切片。  

2. 生成 JSONL 格式的 chunks 文件，每行一个 chunk。  

3. 返回 chunks 文件路径。

  

**典型用法**  

  

```python

from pathlib import Path

from qwen_rag_eval.dataset_tools.chunking import make_chunks_from_samples

  

if __name__ == "__main__":

    samples_path = Path("datasets/processed/my_samples.json")

    chunks_path = Path("datasets/processed/my_samples_chunks.jsonl")

    make_chunks_from_samples(samples_path, chunks_path)

```

  

### 6. 加载 CMRC 评估样本

  

**函数**  

  

```python

from qwen_rag_eval.dataset_tools.loader import load_cmrc_samples

  

def load_cmrc_samples(

    config: YamlConfigReader | str = "config/application.yaml",

) -> list[dict]: ...

```

  

**功能说明**  

  

1. 按配置读取 `dataset.samples_path` 对应的 JSON 文件。  

2. 返回样本列表，不做任何额外加工。  

3. 适合作为评估阶段的统一数据入口。

  

**典型用法**  

  

```python

from qwen_rag_eval.dataset_tools.loader import load_cmrc_samples

  

if __name__ == "__main__":

    samples = load_cmrc_samples("config/application.yaml")

    print("样本数量:", len(samples))

```

  

## 二、数据规范

  

本节约定了 samples 文件与 chunks 文件的字段格式。只要数据满足以下规范，即可直接配合上述接口使用。

  

### 1. 评估样本文件（samples）

  

1. 文件形式  

  

    1\) 文件类型为 JSON，顶层为列表。  

    2\) 默认路径由 `dataset.samples_path` 指定，例如  

       `datasets/processed/cmrc2018_samples.json`。  

    3\) 每个元素为一个样本对象。

  

2. 单条样本的字段规范  

  

    每个样本推荐包含以下字段：  

  

    1\) `id`（必填）  

       类型：字符串。  

       含义：样本唯一标识，在整个文件中应保持唯一。  

  

    2\) `question`（建议）  

       类型：字符串。  

       含义：用于评估的提问内容。  

  

    3\) `ground_truth`（建议）  

       类型：字符串。  

       含义：标准答案文本，用于评估指标计算。  

  

    4\) `ground_truth_context`（必填）  

       类型：字符串。  

       含义：包含标准答案的完整上下文文本，用于切片构建向量库。  

  

3. 样例

  

```json

[

  {

    "id": "CMRC_00001_Q1",

    "question": "《战国无双3》是由哪两个公司合作开发的？",

    "ground_truth": "由光荣和ω-Force合作开发。",

    "ground_truth_context": "……《战国无双3》是由光荣和ω-Force合作开发的一款动作游戏……"

  },

  {

    "id": "CMRC_00001_Q2",

    "question": "游戏的发行公司是哪一家？",

    "ground_truth": "光荣公司。",

    "ground_truth_context": "……该作由光荣公司发行……"

  }

]

```

  

4. 最小要求与用途说明  

  

    1\) 仅构建向量库时：必须存在 `id` 与 `ground_truth_context`。  

    2\) 需要 RAG 评估时：应同时提供 `question` 与 `ground_truth`。  

  

### 2. chunks 文件

  

1. 文件形式  

  

    1\) 文件类型为 JSON Lines（JSONL），每行一个 JSON 对象。  

    2\) 默认路径由 `dataset.chunks_path` 指定，例如  

       `datasets/processed/cmrc2018_chunks.jsonl`。  

    3\) 调用 `build_vector_store_from_samples` 时，会在 samples 同目录下生成：  

       `<样本文件名去扩展名>_chunks.jsonl`。

  

2. 单条 chunk 的字段规范  

  

    1\) `doc_id`（必填）  

       类型：字符串。  

       含义：chunk 唯一 ID，默认格式为 `<样本 id>_<序号>`，例如 `CMRC_00001_Q1_0`。  

  

    2\) `sample_id`（必填）  

       类型：字符串。  

       含义：对应的样本 id，应与 samples 中的 `id` 一致。  

  

    3\) `text`（必填）  

       类型：字符串。  

       含义：该 chunk 的实际文本内容，用于向量化与检索。  

  

3. 样例

  

```json

{"doc_id": "CMRC_00001_Q1_0", "sample_id": "CMRC_00001_Q1", "text": "《战国无双3》是由光荣和ω-Force合作开发的一款动作游戏。"}

{"doc_id": "CMRC_00001_Q1_1", "sample_id": "CMRC_00001_Q1", "text": "本作在角色数量与系统上……"}

{"doc_id": "CMRC_00001_Q2_0", "sample_id": "CMRC_00001_Q2", "text": "该作由光荣公司发行……"}

```

  

4. 与向量库的对应关系  

  

    1\) `convert_chunks_to_documents` 会将每条记录转换为 `Document`：  

       `page_content = text`，`metadata = {"doc_id": ..., "sample_id": ...}`。  

    2\) `VectorStoreManager.add_documents` 直接写入 Chroma collection。  

  

### 3. 关键配置项

  

在 `config/application.yaml` 中，至少与本 README 相关的配置如下：

  

```yaml

dataset:

  raw_path: "datasets/raw/cmrc2018_raw.json"          # 仅在从原始数据构建时使用

  samples_path: "datasets/processed/cmrc2018_samples.json"

  chunks_path: "datasets/processed/cmrc2018_chunks.jsonl"

  sample_limit: 200                                   # 可选，raw→samples 抽样数量

  

vector_store:

  embedding_model: "text-embedding-v4"

  persist_directory: "chroma_db/cmrc2018"

  collection_name: "cmrc2018"

```

  

说明：  

  

1. `dataset.samples_path` 与 `dataset.chunks_path` 决定了数据文件的默认位置。  

2. `vector_store.*` 决定了向量库所使用的 embedding 模型、持久化目录与默认集合名。  

3. 若仅使用 `build_vector_store_from_samples`，则只需保证 samples 文件与 `vector_store.*` 配置正确，无需提供 raw 数据。