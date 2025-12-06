# qwen_rag_eval/dataset_tools/sampling.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils import YamlConfigReader


def _get_project_root(config: YamlConfigReader) -> Path:
    """
    约定：application.yaml 位于 project-root/config/ 目录下，
    因此 project-root = config_path.parent.parent
    """
    return config.config_path.parent.parent


def load_raw_dataset(config: YamlConfigReader) -> List[Dict[str, Any]]:
    """
    从 dataset.raw_path 加载原始 CMRC 数据。
    """
    project_root = _get_project_root(config)

    raw_rel = config.get("dataset.raw_path")
    if not raw_rel:
        raise ValueError("配置缺少 dataset.raw_path")

    raw_file = project_root / raw_rel
    if not raw_file.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{raw_file}")

    with raw_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def cmrc_to_rag_format(
    raw_data: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    将 CMRC 原始结构转换为 RAG 评估样本结构：
      { id, question, ground_truth, ground_truth_context }
    """
    rows: List[Dict[str, Any]] = []

    for item in raw_data:
        ctx = item["context_text"]

        for qa in item["qas"]:
            rows.append(
                {
                    "id": qa["query_id"],
                    "question": qa["query_text"],
                    "ground_truth": qa["answers"][0],
                    "ground_truth_context": ctx,
                }
            )

    if limit is not None:
        rows = rows[:limit]

    return rows


def build_eval_samples(
    config: Union[YamlConfigReader, str] = "config/application.yaml",
) -> Path:
    """
    从原始 CMRC 数据构建评估样本（samples），写入 dataset.samples_path。

    如果目标文件已存在，则直接返回路径。
    """
    if isinstance(config, str):
        config = YamlConfigReader(config)

    project_root = _get_project_root(config)

    samples_rel = (
        config.get("dataset.samples_path")
        or "datasets/processed/cmrc2018_samples.json"
    )
    sample_limit = config.get("dataset.sample_limit") or 200

    samples_file = project_root / samples_rel
    if samples_file.exists():
        print(f"[sampling] 样本已存在：{samples_file}")
        return samples_file

    raw_data = load_raw_dataset(config)
    rows = cmrc_to_rag_format(raw_data, limit=sample_limit)

    samples_file.parent.mkdir(parents=True, exist_ok=True)
    with samples_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[sampling] 生成样本：{samples_file}，共 {len(rows)} 条")
    return samples_file
