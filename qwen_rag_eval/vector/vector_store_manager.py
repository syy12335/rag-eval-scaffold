# qwen_rag_eval/vector/vector_store_manager.py

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.schema import Document
from chromadb import PersistentClient

from utils import YamlConfigReader  # 顶层 utils 包

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"


class VectorStoreManager:
    """
    VectorStoreManager：用于管理 Chroma 向量库。

    对外接口保持兼容：
      - __init__() 仍然可以无参调用
      - load_or_create_collection(collection_name)
      - add_documents(documents, collection_name)
      - get_retriever(collection_name, k=3)
      - delete_collection(collection_name)
    """

    def __init__(self, config: YamlConfigReader | None = None):
        """
        参数
        config: 可选的 YamlConfigReader 实例。
                如果不传，则默认读取 "config/application.yaml"。
        """
        # 1. 读取配置
        if config is None:
            config = YamlConfigReader("config/application.yaml")
        self.config = config

        # 2. 计算 project_root（约定：config/application.yaml 在 project-root/config 下）
        #    project_root = .../project-root
        self.project_root = self.config.config_path.parent.parent

        # 3. 持久化目录
        #    优先使用新的 vector_store.persist_directory，
        #    回退到旧的 path.embedding-directory，最后给一个默认值。
        persist_rel = (
            self.config.get("vector_store.persist_directory")
            or self.config.get("path.embedding-directory")
            or "chroma_db/cmrc2018"
        )
        self.persist_directory = str(self.project_root / persist_rel)

        # 4. embedding 模型
        #    同样优先使用新的 vector_store.embedding_model，
        #    回退到旧的 embedding.model，再回退到 text-embedding-v4。
        embedding_model = (
            self.config.get("vector_store.embedding_model")
            or self.config.get("embedding.model")
            or "text-embedding-v4"
        )

        api_key = os.environ.get("API_KEY_Qwen")
        if not api_key:
            raise ValueError(
                "未在环境变量 API_KEY_Qwen 中找到千问 API Key，"
                "请先设置：set API_KEY_Qwen=你的key"
            )

        self.embedding = DashScopeEmbeddings(
            model=embedding_model,
            dashscope_api_key=api_key,
        )

        self.vectorstore: Optional[Chroma] = None

    def load_or_create_collection(self, collection_name: str) -> Chroma:
        """
        加载或创建指定 collection。

        如果当前已经持有同名 collection，直接复用；
        否则在同一 persist_directory 下创建 / 加载新的 collection。
        """
        if (
            self.vectorstore is not None
            and getattr(self.vectorstore, "_collection", None) is not None
            and self.vectorstore._collection.name == collection_name
        ):
            return self.vectorstore

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_directory,
        )
        return self.vectorstore

    def add_documents(
        self,
        documents: List[Document],
        collection_name: str,
        batch_size: int = 10,
    ) -> None:
        """
        将一批 Document 写入指定 collection，内部自动分批。

        参数
        documents: langchain.schema.Document 列表
        collection_name: 目标 collection 名称
        batch_size: 每批写入大小
        """
        if not documents:
            return

        vs = self.load_or_create_collection(collection_name)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            print(f"写入文档 {i + 1}-{min(i + batch_size, total)} / {total}")
            vs.add_documents(batch)

    def get_retriever(self, collection_name: str, k: int = 3):
        """
        获取指定 collection 的 retriever。
        """
        vs = self.load_or_create_collection(collection_name)
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def delete_collection(self, collection_name: str) -> None:
        """
        删除指定 collection，如果存在的话。
        """
        client = PersistentClient(path=self.persist_directory)
        existing_collections = [col.name for col in client.list_collections()]

        if collection_name not in existing_collections:
            print(f"集合 '{collection_name}' 不存在，无需删除。")
            return

        client.delete_collection(name=collection_name)
        print(f"已成功删除集合：'{collection_name}'")

        # 同步清理内存中的 vectorstore 引用
        try:
            if self.vectorstore and getattr(self.vectorstore, "_collection", None):
                if getattr(self.vectorstore._collection, "name", None) == collection_name:
                    self.vectorstore = None
        except Exception:
            self.vectorstore = None
