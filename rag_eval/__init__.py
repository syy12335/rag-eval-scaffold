# rag_eval/__init__.py
"""
rag_eval 顶层包的公共 API。

当前稳定暴露的核心组件包括：

1. VectorDatabaseBuilder
   用于从评估样本（samples）与 chunk 文件构建向量库。
   典型用法：
       from rag_eval import VectorDatabaseBuilder
       builder = VectorDatabaseBuilder("config/application.yaml")
       manager = builder.invoke(overwrite=True)

   invoke 输入参数：
       samples_path: 可选。显式指定 samples.json 路径。
                     若为 None，则从 application.yaml 的 dataset.samples_path 读取。
       collection_name: 可选。显式指定向量库 collection 名称。
                        若为 None，则从 application.yaml 的 vector_store.collection_name 读取。
       overwrite: bool。当目标 collection 已存在时是否覆盖。

   invoke 返回值：
       已完成构建的 VectorStoreManager 实例，可继续通过 get_retriever 进行检索。

2. dataset_tools
   数据集适配层，目前内置 cmrc2018 子模块：
       from rag_eval.dataset_tools import cmrc2018
       samples_path = cmrc2018.build_eval_samples(config)
"""

from .vector.vector_builder import VectorDatabaseBuilder
from . import dataset_tools

__all__ = [
    "VectorDatabaseBuilder",
    "dataset_tools",
]
