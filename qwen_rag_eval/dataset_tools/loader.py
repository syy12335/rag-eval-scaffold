# qwen_rag_eval/dataset_tools/loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from utils import YamlConfigReader


def _get_project_root(config: YamlConfigReader) -> Path:
    return config.config_path.parent.parent


def load_cmrc_samples(
    config: Union[YamlConfigReader, str] = "config/application.yaml",
) -> List[Dict[str, Any]]:
    """
    统一的 CMRC 评估样本加载接口。

    只读 dataset.samples_path 中的 JSON 文件，不做任何加工。
    """
    if isinstance(config, str):
        config = YamlConfigReader(config)

    project_root = _get_project_root(config)

    samples_rel = config.get("dataset.samples_path")
    if not samples_rel:
        raise ValueError("配置缺少 dataset.samples_path")

    samples_file = project_root / samples_rel
    if not samples_file.exists():
        raise FileNotFoundError(f"未找到评估样本文件：{samples_file}")

    with samples_file.open("r", encoding="utf-8") as f:
        return json.load(f)
