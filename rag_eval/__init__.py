# rag_eval/__init__.py
"""
rag_eval 顶层包的公共 API。

当前稳定暴露的核心组件包括：

1. VectorDatabaseBuilder
   用途：
       从评估样本（samples）与 chunk 文件构建向量库。

   典型用法：
       from rag_eval import VectorDatabaseBuilder
       builder = VectorDatabaseBuilder("config/application.yaml")
       manager = builder.invoke(overwrite=True)

   invoke 参数约定：
       samples_path:
           可选。显式指定 samples.json 路径。
           若为 None，则从 application.yaml 的 dataset.samples_path 读取。
       collection_name:
           可选。显式指定向量库 collection 名称。
           若为 None，则从 application.yaml 的 vector_store.collection_name 读取。
       overwrite:
           bool。当目标 collection 已存在时是否覆盖。

   返回值：
       已完成构建的 VectorStoreManager 实例，可继续通过 get_retriever 进行检索。

2. dataset_tools
   用途：
       数据集适配层，目前内置 cmrc2018 子模块。

   典型用法：
       from rag_eval.dataset_tools import cmrc2018
       samples_path = cmrc2018.build_eval_samples(config)

3. RagRunner
   用途：
       基于已有向量库运行一条默认的 RAG 工作流（检索 + 生成），
       内部会根据 config/application.yaml 与 config/model_roles.yaml 的 generation 段构造 LLM 和提示词。

   典型用法：
       from rag_eval import VectorDatabaseBuilder, RagRunner

       # 1) 构建向量库
       builder = VectorDatabaseBuilder("config/application.yaml")
       vector_manager = builder.invoke(overwrite=False)

       # 2) 基于向量库创建 Runner
       runner = RagRunner.from_vector_store(
           vector_manager,
           config_path="config/application.yaml",
       )

       # 3) 执行一次 RAG
       result = runner.invoke("这里写你的问题")
       # result 结构示例：
       # {
       #   "question": str,
       #   "answer": str,
       #   "contexts": [
       #       {"content": str, "metadata": dict},
       #       ...
       #   ]
       # }
"""

from .vector.vector_builder import VectorDatabaseBuilder
from . import dataset_tools
from .rag import RagRunner

__all__ = [
    "VectorDatabaseBuilder",
    "dataset_tools",
    "RagRunner",
]
