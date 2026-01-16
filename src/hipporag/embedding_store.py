import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"创建工作目录: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        插入文本字符串并编码为embeddings

        当OOM导致需要重新执行时，会自动重试整个流程
        """
        max_retries = 3  # 最多重试3次整个流程
        retry_count = 0

        while retry_count <= max_retries:
            try:
                self._insert_strings_impl(texts)
                return  # 成功完成
            except RuntimeError as e:
                if "连续5次OOM" in str(e) and "需要重新执行" in str(e):
                    # 5连续OOM时，先保存当前内存中已完成的数据防止丢失
                    logger.warning(f"⚠️ 检测到5连续OOM，立即保存已完成的数据（共{len(self.hash_ids)}条）...")
                    self._save_data()
                    logger.info(f"✓ 已完成的{len(self.hash_ids)}条数据已保存到缓存")

                    # 经验：同一进程内的CUDA/bitsandbytes状态很难彻底恢复；最稳妥是直接重启进程断点续跑。
                    import os
                    import sys

                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"已重试{max_retries}次，仍然失败，程序终止")
                        raise

                    restart_enabled = os.getenv("HIPPORAG_OOM_RESTART", "1").lower() not in ("0", "false", "no")
                    if not restart_enabled:
                        logger.error("自动重启已禁用（设置 HIPPORAG_OOM_RESTART=0），请手动重新执行程序")
                        raise

                    try:
                        max_restarts = int(os.getenv("HIPPORAG_OOM_RESTART_MAX", "3"))
                    except Exception:
                        max_restarts = 3

                    try:
                        restart_count = int(os.getenv("HIPPORAG_OOM_RESTART_COUNT", "0"))
                    except Exception:
                        restart_count = 0

                    if restart_count >= max_restarts:
                        logger.error(f"已达到最大自动重启次数 {restart_count}/{max_restarts}，请手动重新执行程序")
                        raise

                    os.environ["HIPPORAG_OOM_RESTART_COUNT"] = str(restart_count + 1)
                    logger.error(f"准备自动重启进程以清空CUDA/bnb状态 ({restart_count + 1}/{max_restarts})...")
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    raise

    def _insert_strings_impl(self, texts: List[str]):
        """
        实际的插入实现逻辑
        """
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"插入 {len(missing_ids)} 条新记录，{len(all_hash_ids) - len(missing_ids)} 条记录已存在")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 分批编码和保存，每1000条保存一次，支持断点续传（减小chunk_size以降低GPU内存压力）
        chunk_size = 1000
        for i in range(0, len(texts_to_encode), chunk_size):
            chunk_texts = texts_to_encode[i:i + chunk_size]
            chunk_ids = missing_ids[i:i + chunk_size]

            logger.info(f"编码批次 {i//chunk_size + 1}/{(len(texts_to_encode) + chunk_size - 1)//chunk_size} ({len(chunk_texts)} 条记录)")

            try:
                chunk_embeddings = self.embedding_model.batch_encode(chunk_texts)
                # 立即保存这一批
                self._upsert(chunk_ids, chunk_texts, chunk_embeddings)
                # 可选：每个chunk后做一次CUDA清理，降低碎片化/后续chunk突然OOM的概率
                import os
                if os.getenv("HIPPORAG_EMBED_CHUNK_CUDA_CLEANUP", "1").lower() not in ("0", "false", "no"):
                    try:
                        import gc
                        import torch
                        import time
                        for _ in range(3):
                            gc.collect()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        # 给驱动一点时间回收/整理（避免下一个chunk开头free=0）
                        time.sleep(float(os.getenv("HIPPORAG_EMBED_CHUNK_CUDA_SLEEP", "0.2")))
                    except Exception:
                        pass
            except RuntimeError as e:
                if "连续5次OOM" in str(e) and "需要重新执行" in str(e):
                    # OOM失败，检查是否有已编码的部分数据
                    if hasattr(e, 'partial_results') and e.partial_results is not None:
                        # 有部分编码结果，保存这部分（支持乱序/非前缀）
                        partial_count = len(e.partial_results)
                        logger.warning(f"⚠️ OOM时有{partial_count}条已编码数据，立即保存...")

                        if hasattr(e, "partial_indices") and e.partial_indices:
                            partial_ids = [chunk_ids[i] for i in e.partial_indices]
                            partial_texts = [chunk_texts[i] for i in e.partial_indices]
                        else:
                            partial_ids = chunk_ids[:partial_count]
                            partial_texts = chunk_texts[:partial_count]

                        self._upsert(partial_ids, partial_texts, e.partial_results)
                        logger.info(f"✓ 已保存{partial_count}条部分编码数据")
                    # OOM失败，重新抛出给上层的insert_strings处理
                    raise
                else:
                    logger.error(f"编码批次 {i//chunk_size + 1} 失败: {str(e)}")
                    logger.info(f"已成功保存前 {i} 条记录，可以从此处继续")
                    raise

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"从文件加载了 {len(self.hash_ids)} 条记录: {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"已保存 {len(self.hash_ids)} 条记录到 {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"保存新记录")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
