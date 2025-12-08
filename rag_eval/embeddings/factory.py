# rag_eval/embeddings/factory.py

from __future__ import annotations

import os
from typing import Final

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings

from utils import YamlConfigReader


def build_embedding_from_config(config: YamlConfigReader) -> Embeddings:
    """
    根据 embedding.* 配置构造一个 Embeddings 对象。

    约定的配置 schema：

        embedding:
          provider: "dashscope"
          model: "text-embedding-v4"
          api_key_env: "API_KEY_Qwen"

    行为：

        1）不提供任何默认值，缺什么字段就抛错，让配置成为唯一真值。
        2）provider == "dashscope" 时，使用 DashScopeEmbeddings。
        3）其他 provider 暂不支持，抛出明确错误，避免静默 fallback。
    """
    provider = config.get("embedding.provider")
    if not provider:
        raise ValueError("配置缺少 embedding.provider")

    model = config.get("embedding.model")
    if not model:
        raise ValueError("配置缺少 embedding.model")

    api_key_env = config.get("embedding.api_key_env")
    if not api_key_env:
        raise ValueError("配置缺少 embedding.api_key_env")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"[embedding_factory] 未在环境变量 {api_key_env} 中找到 API Key，"
            f"请先设置环境变量 {api_key_env}"
        )

    provider = provider.lower()

    if provider == "dashscope":
        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
        )

    raise ValueError(
        f"[embedding_factory] 暂不支持 embedding.provider = {provider!r}，"
        "请先在 factory 中添加对应实现再使用该 provider"
    )
