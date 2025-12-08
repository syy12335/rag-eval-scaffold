# qwen_rag_eval/dataset_tools/chunking.py

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter


def make_chunks_from_samples(
    samples_path: Union[str, Path],
    chunks_path: Union[str, Path],
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> Path:
    """
    将 samples 中的 ground_truth_context 切片，输出为 JSONL chunks 文件。

    输入：
      samples_path: 含 question / ground_truth / ground_truth_context 的 JSON 文件
      chunks_path:  输出 JSONL 文件路径（每行一个 chunk 记录）
    """
    samples_path = Path(samples_path)
    chunks_path = Path(chunks_path)

    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " "],
    )

    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    with chunks_path.open("w", encoding="utf-8") as f_out:
        for s in samples:
            ctx = s["ground_truth_context"]
            sid = s["id"]

            chunks = splitter.split_text(ctx)
            for i, text in enumerate(chunks):
                record = {
                    "doc_id": f"{sid}_{i}",
                    "sample_id": sid,
                    "text": text,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[chunking] 从 samples 切割完成：{samples_path} → {chunks_path}，"
        f"共 {len(samples)} 条样本"
    )
    return chunks_path


def load_chunk_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    从 JSONL chunks 文件中读取所有记录（dict 列表）。
    """
    path = Path(path)
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records
