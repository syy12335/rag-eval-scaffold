# rag_eval/vector/vector_store_manager.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from chromadb import PersistentClient
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from utils import YamlConfigReader
from rag_eval.embeddings.factory import build_embedding_from_config


class VectorStoreManager:
    """
    VectorStoreManager：管理 Chroma 向量库。

    只负责向量库本身：
      1）persist_directory
      2）collection 的创建 / 删除 / 复用
      3）写入 Document、返回 retriever

    Embedding 完全通过构造函数注入，或由 embedding_factory 统一创建，
    不在这里出现任何 provider / 模型名。
    """

    def __init__(
        self,
        config: YamlConfigReader | None = None,
        embedding: Optional[Embeddings] = None,
    ) -> None:
        if config is None:
            config = YamlConfigReader("config/application.yaml")
        self.config = config

        # project_root：config/application.yaml 位于 project-root/config
        self.project_root: Path = self.config.config_path.parent.parent

        # 向量库持久化目录（只看 vector_store.persist_directory）
        persist_rel = self.config.get("vector_store.persist_directory")
        if not persist_rel:
            raise ValueError("配置缺少 vector_store.persist_directory")

        self.persist_directory: str = str(self.project_root / persist_rel)

        # Embedding 后端
        if embedding is not None:
            self.embedding: Embeddings = embedding
        else:
            self.embedding = build_embedding_from_config(self.config)

        self.vectorstore: Optional[Chroma] = None

    def load_or_create_collection(self, collection_name: str) -> Chroma:
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
        if not documents:
            return

        vs = self.load_or_create_collection(collection_name)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            print(
                f"[vector_store_manager] 写入文档 "
                f"{i + 1}-{min(i + batch_size, total)} / {total}"
            )
            vs.add_documents(batch)

    def get_retriever(self, collection_name: str, k: int = 3):
        vs = self.load_or_create_collection(collection_name)
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def delete_collection(self, collection_name: str) -> None:
        client = PersistentClient(path=self.persist_directory)
        existing_collections = [col.name for col in client.list_collections()]

        if collection_name not in existing_collections:
            print(f"[vector_store_manager] 集合 '{collection_name}' 不存在，无需删除。")
            return

        client.delete_collection(name=collection_name)
        print(f"[vector_store_manager] 已成功删除集合：'{collection_name}'")

        if (
            self.vectorstore
            and getattr(self.vectorstore, "_collection", None)
            and getattr(self.vectorstore._collection, "name", None) == collection_name
        ):
            self.vectorstore = None
