from copy import deepcopy
from typing import List, Optional
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


class NVEmbedV2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_dim = self.embedding_model.config.hidden_size
        self._adaptive_state = {}



    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        # 检查是否需要量化
        use_quantization = getattr(self.global_config, 'use_quantization', False)
        quantization_bits = getattr(self.global_config, 'quantization_bits', 4)

        model_init_params = {
            "pretrained_model_name_or_path": self.embedding_model_name,
            "trust_remote_code": True,
            'device_map': "auto",
            "torch_dtype": self.global_config.embedding_model_dtype,
        }

        # 如果启用量化，添加量化配置
        if use_quantization:
            logger.info(f"启用 {quantization_bits}-bit 量化加载 NV-Embed-v2")

            if quantization_bits == 4:
                # 4-bit NF4量化配置（推荐）
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # Normal Float 4
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,  # 双重量化
                )
                model_init_params["quantization_config"] = bnb_config
                logger.info("使用4-bit NF4量化，预计显存占用: ~5.5GB")

            elif quantization_bits == 8:
                # 8-bit量化配置
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_init_params["quantization_config"] = bnb_config
                logger.info("使用8-bit量化，预计显存占用: ~7.8GB")
            else:
                logger.warning(f"不支持的量化位数: {quantization_bits}，将使用FP16/BF16")
        else:
            logger.info("未启用量化，使用FP16/BF16加载（需要~15.6GB显存）")

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": model_init_params,
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        # 注意：不打印包含BitsAndBytesConfig的配置，因为它不能被JSON序列化
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config (quantization={use_quantization})")

    def _load_model(self) -> None:
        """
        重新加载embedding model，用于在OOM后恢复GPU内存
        """
        logger.warning("重新加载embedding model...")
        import gc

        # 在重新加载前尽可能释放旧模型占用的显存
        try:
            if getattr(self, "embedding_model", None) is not None:
                del self.embedding_model
        except Exception:
            pass

        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        # 保持与初次加载一致的参数（device_map/量化配置等），避免同进程重载行为不一致
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_dim = self.embedding_model.config.hidden_size
        logger.info("✓ embedding model重新加载完成")

    def unload_model(self) -> None:
        """
        尽可能释放embedding model占用的GPU/CPU内存。
        注意：CUDA上下文/第三方库(bnb)可能仍会保留部分缓存，彻底释放通常需要重启进程。
        """
        import gc

        try:
            if getattr(self, "embedding_model", None) is not None:
                del self.embedding_model
        except Exception:
            pass

        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        动态调节batch size的编码函数

        逻辑：
        - 初始batch_size=4
        - 每成功处理100条记录，batch_size+1（最大12）
        - 发生OOM时，先清理GPU碎片，再降低batch_size为当前值的一半（最小1）
        - 只有连续5次batch_size=1时都OOM才结束程序
        """
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "

        # 移除原始batch_size参数，使用动态调节
        params.pop("batch_size", None)

        # 动态batch size参数
        current_batch_size = 6  # 初始batch size
        min_batch_size = 1  # 最小batch size
        success_count = 0  # 成功处理的记录数
        increase_threshold = 50  # 改为每处理50条增加batch size
        oom_count_at_min = 0  # 在batch_size=1时连续OOM的次数
        max_oom_retries = 5  # 最大连续OOM重试次数
        import os
        debug = os.getenv("HIPPORAG_EMBED_DEBUG", "0").lower() not in ("0", "false", "no")
        token_budget = int(os.getenv("HIPPORAG_EMBED_TOKEN_BUDGET", "0"))
        auto_tune = os.getenv("HIPPORAG_EMBED_AUTO_TUNE", "1").lower() not in ("0", "false", "no")

        # Heuristic defaults by GPU size to avoid repeated OOM without requiring env vars.
        total_mem_gb = None
        if torch.cuda.is_available():
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                total_mem_gb = None
        default_max_batch_size = 8 if (total_mem_gb is not None and total_mem_gb <= 9.5) else 12
        max_batch_size = int(os.getenv("HIPPORAG_EMBED_MAX_BATCH_SIZE", str(default_max_batch_size)))

        # auto-tune knobs
        tune_cooldown_batches = int(os.getenv("HIPPORAG_EMBED_TUNE_COOLDOWN", "12"))
        tune_up_batches = int(os.getenv("HIPPORAG_EMBED_TUNE_UP_EVERY", "6"))
        cap_up_batches = int(os.getenv("HIPPORAG_EMBED_CAP_UP_EVERY", "30"))
        token_budget_min = int(os.getenv("HIPPORAG_EMBED_TOKEN_BUDGET_MIN", "256"))
        token_budget_max = int(os.getenv("HIPPORAG_EMBED_TOKEN_BUDGET_MAX", "8192"))
        token_budget_up_pct = float(os.getenv("HIPPORAG_EMBED_TOKEN_BUDGET_UP_PCT", "0.10"))
        token_budget_down_pct = float(os.getenv("HIPPORAG_EMBED_TOKEN_BUDGET_DOWN_PCT", "0.20"))

        # speed-tune: 如果增大batch_size反而更慢，则自动把上限锁回更快的batch_size
        speed_tune = os.getenv("HIPPORAG_EMBED_SPEED_TUNE", "1").lower() not in ("0", "false", "no")
        speed_drop_pct = float(os.getenv("HIPPORAG_EMBED_SPEED_DROP_PCT", "0.10"))  # 下降超过10%则回退
        speed_test_batches = int(os.getenv("HIPPORAG_EMBED_SPEED_TEST_BATCHES", "3"))  # 新bs跑几次再判断
        speed_window = int(os.getenv("HIPPORAG_EMBED_SPEED_WINDOW", "6"))  # 计算当前bs平均速度的窗口

        # Persistent (per-process) tuned ceiling so we don't repeatedly climb into known-OOM batch sizes.
        bs_cap = int(self._adaptive_state.get("bs_cap", max_batch_size))
        bs_cap = max(min_batch_size, min(max_batch_size, bs_cap))
        start_bs = self._adaptive_state.get("start_bs", None)
        if start_bs is None:
            current_batch_size = min(bs_cap, 4)
        else:
            current_batch_size = max(min_batch_size, min(bs_cap, int(start_bs)))

        logger.info(f"开始动态batch size编码，初始batch_size={current_batch_size} (cap={bs_cap})")

        pbar = tqdm(total=len(texts), desc="Batch Encoding")
        outputs: List[Optional[np.ndarray]] = [None] * len(texts)
        processed_indices: List[int] = []

        import gc

        def _maybe_length_bucket_indices() -> tuple[List[int], Optional[List[int]]]:
            """
            不改变输入内容的情况下，通过按token长度分桶减少padding浪费，从而降低OOM概率。
            输出为处理顺序的索引列表；最终会按原顺序写回 `outputs`，不影响结果对齐。
            """
            import os

            enabled = os.getenv("HIPPORAG_EMBED_LENGTH_BUCKET", "1").lower() not in ("0", "false", "no")
            if not enabled or len(texts) <= 1:
                return list(range(len(texts))), None

            max_length = params.get("max_length", None)
            instruction = params.get("instruction", "") or ""

            try:
                from transformers import AutoTokenizer

                tokenizer = getattr(self, "_tokenizer", None)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.embedding_model_name,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                    setattr(self, "_tokenizer", tokenizer)

                # 注意：估算长度用于“减少padding”，不改变 encode() 实际输入。
                # 这里使用 truncation=True 仅用于估计“encode阶段最多会用到的长度上限”，不会改变实际传入文本。
                enc = tokenizer(
                    [instruction + t for t in texts],
                    add_special_tokens=True,
                    truncation=True if isinstance(max_length, int) else False,
                    max_length=max_length if isinstance(max_length, int) else None,
                    padding=False,
                    return_length=True,
                )
                lengths = enc.get("length", None)
                if lengths is None:
                    input_ids = enc.get("input_ids", [])
                    lengths = [len(x) for x in input_ids]

                # 按长度升序排序，batch内长度更接近 -> padding更少（不改变文本，不影响精度）
                # 升序有利于先跑短文本以快速升速；长文本会在后面自动回退预算/批大小。
                order = sorted(range(len(texts)), key=lambda i: lengths[i])
                return order, list(lengths)
            except Exception:
                return list(range(len(texts))), None

        def _cuda_mem_gb() -> Optional[tuple[float, float, float]]:
            if not torch.cuda.is_available():
                return None
            try:
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free, total = torch.cuda.mem_get_info()
                free_gb = free / 1024**3
                return alloc, reserved, free_gb
            except Exception:
                return None

        def _cleanup_cuda(tag: str) -> None:
            if not torch.cuda.is_available():
                return

            before = _cuda_mem_gb()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            for _ in range(2):
                gc.collect()

            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            after = _cuda_mem_gb()
            if before and after:
                logger.warning(
                    f"{tag} CUDA内存: alloc {before[0]:.2f}→{after[0]:.2f}GB, "
                    f"reserved {before[1]:.2f}→{after[1]:.2f}GB, free {before[2]:.2f}→{after[2]:.2f}GB"
                )

        order, lengths_for_debug = _maybe_length_bucket_indices()
        # 预清理：避免“新chunk刚开始时 free≈0，首个batch直接OOM”
        if os.getenv("HIPPORAG_EMBED_PREFLIGHT_CLEANUP", "1").lower() not in ("0", "false", "no"):
            _cleanup_cuda("batch_encode开始前清理")
        pos = 0
        oom_cooldown_left = 0
        stable_batches = 0
        stable_batches_for_cap = 0

        # speed tracking (per-call; decision state persisted in self._adaptive_state)
        speed_by_bs: dict[int, List[float]] = {}
        speed_test_state = self._adaptive_state.get("speed_test_state", None) or {}

        while pos < len(order):
            # 确定当前批次大小
            if token_budget > 0 and lengths_for_debug is not None:
                # token预算batch：在不截断的前提下尽量避免“某一批突然变长导致OOM”
                remaining = len(order) - pos
                max_take = min(current_batch_size, remaining)
                batch_indices = []
                used = 0
                for j in order[pos:pos + max_take]:
                    l = int(lengths_for_debug[j])
                    if batch_indices and used + l > token_budget:
                        break
                    batch_indices.append(j)
                    used += l
                if not batch_indices:
                    batch_indices = [order[pos]]
                actual_batch_size = len(batch_indices)
            else:
                actual_batch_size = min(current_batch_size, len(order) - pos)
                batch_indices = order[pos:pos + actual_batch_size]

            params["prompts"] = [texts[j] for j in batch_indices]

            try:
                # 轻量清理：释放PyTorch缓存（不保证解决真实OOM，只减少瞬时峰值/碎片影响）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if debug and lengths_for_debug is not None:
                    blens = [lengths_for_debug[j] for j in batch_indices]
                    mem = _cuda_mem_gb()
                    if mem:
                        alloc_gb, reserved_gb, free_gb = mem
                        logger.info(
                            f"[embed-debug] bs={current_batch_size} actual={actual_batch_size} "
                            f"len_max={max(blens)} len_mean={sum(blens)/len(blens):.1f} "
                            f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB free={free_gb:.2f}GB"
                        )
                    else:
                        logger.info(
                            f"[embed-debug] bs={current_batch_size} actual={actual_batch_size} "
                            f"len_max={max(blens)} len_mean={sum(blens)/len(blens):.1f}"
                        )

                # 编码当前批次
                t0 = time.perf_counter()
                with torch.inference_mode():
                    batch_result = self.embedding_model.encode(**params)
                dt = max(1e-6, time.perf_counter() - t0)
                ips = float(actual_batch_size) / dt
                speed_by_bs.setdefault(int(current_batch_size), []).append(ips)
                if len(speed_by_bs[int(current_batch_size)]) > speed_window:
                    speed_by_bs[int(current_batch_size)] = speed_by_bs[int(current_batch_size)][-speed_window:]

                # 关键：立刻把结果搬到CPU，避免在同一进程中累积GPU tensor导致碎片化/显存占用爬升
                if isinstance(batch_result, torch.Tensor):
                    batch_result = batch_result.detach().to("cpu")
                    if batch_result.dtype == torch.bfloat16:
                        batch_result = batch_result.to(torch.float32)
                    batch_result = batch_result.numpy()

                for local_i, original_i in enumerate(batch_indices):
                    outputs[original_i] = batch_result[local_i]
                    processed_indices.append(original_i)

                # 成功处理，重置OOM计数器
                oom_count_at_min = 0
                stable_batches += 1
                stable_batches_for_cap += 1
                if oom_cooldown_left > 0:
                    oom_cooldown_left -= 1

                # 更新进度
                pos += actual_batch_size
                success_count += actual_batch_size
                pbar.update(actual_batch_size)

                # 每成功处理50条，增加batch size（有上限）
                if success_count >= increase_threshold:
                    old_bs = int(current_batch_size)
                    if oom_cooldown_left <= 0:
                        current_batch_size = min(bs_cap, current_batch_size + 1)
                    success_count = 0  # 重置计数器
                    logger.info(f"✓ 成功处理50条记录，增加batch_size到 {current_batch_size}")

                    # speed test state: 记录“从old_bs提到new_bs”的对比基线
                    if speed_tune and int(current_batch_size) != old_bs:
                        prev_speeds = speed_by_bs.get(old_bs, [])
                        baseline = float(sum(prev_speeds) / len(prev_speeds)) if prev_speeds else None
                        speed_test_state = {
                            "baseline_bs": old_bs,
                            "baseline_ips": baseline,
                            "test_bs": int(current_batch_size),
                            "test_ips": [],
                        }

                # auto-tune：持续稳定后，温和上调token预算/批大小以提速（不改变文本，不影响精度）
                if auto_tune and lengths_for_debug is not None and stable_batches >= tune_up_batches:
                    stable_batches = 0
                    if oom_cooldown_left <= 0:
                        if token_budget > 0:
                            new_budget = int(token_budget * (1.0 + token_budget_up_pct))
                            token_budget = min(token_budget_max, max(token_budget_min, new_budget))
                        # 如果没启用token_budget，则主要靠batch_size上调（由上面的increase_threshold触发）
                        if debug and token_budget > 0:
                            logger.info(f"[embed-tune] increase token_budget -> {token_budget}")

                # auto-tune：长时间稳定后，允许抬高 batch_size 上限，逐步逼近最大吞吐
                if auto_tune and stable_batches_for_cap >= cap_up_batches and bs_cap < max_batch_size:
                    stable_batches_for_cap = 0
                    if oom_cooldown_left <= 0:
                        bs_cap += 1
                        if debug:
                            logger.info(f"[embed-tune] increase bs_cap -> {bs_cap}")

                # speed-tune：判断提升batch_size是否“变慢太多”，如果是则锁回更快的bs
                if speed_tune and speed_test_state:
                    test_bs = speed_test_state.get("test_bs", None)
                    if test_bs is not None and int(current_batch_size) == int(test_bs):
                        speed_test_state.setdefault("test_ips", []).append(ips)
                        if len(speed_test_state["test_ips"]) >= speed_test_batches:
                            baseline_ips = speed_test_state.get("baseline_ips", None)
                            baseline_bs = speed_test_state.get("baseline_bs", None)
                            test_ips_avg = float(sum(speed_test_state["test_ips"]) / len(speed_test_state["test_ips"]))
                            if baseline_ips is not None and baseline_bs is not None:
                                if test_ips_avg < baseline_ips * (1.0 - speed_drop_pct):
                                    bs_cap = min(bs_cap, int(baseline_bs))
                                    current_batch_size = min(current_batch_size, bs_cap)
                                    oom_cooldown_left = max(oom_cooldown_left, tune_cooldown_batches)
                                    logger.warning(
                                        f"[embed-speed] batch_size={test_bs} 平均{test_ips_avg:.2f}it/s 低于 "
                                        f"batch_size={baseline_bs} 的{baseline_ips:.2f}it/s（>{speed_drop_pct*100:.0f}%），"
                                        f"将bs_cap锁定为 {bs_cap}"
                                    )
                            speed_test_state = {}

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    # OOM错误处理：先清理碎片，再降低batch size
                    if lengths_for_debug is not None:
                        blens = [lengths_for_debug[j] for j in batch_indices]
                        logger.warning(
                            f"✗ 检测到OOM(len_max={max(blens)}, len_mean={sum(blens)/len(blens):.1f})，开始清理GPU碎片..."
                        )
                    else:
                        logger.warning(f"✗ 检测到OOM，开始清理GPU碎片...")

                    # 第一步：清理GPU碎片
                    _cleanup_cuda("OOM清理后")

                    # 第二步：仅在“疑似碎片化/缓存占用较大”时才用原batch_size重试，避免无意义重复OOM
                    mem = _cuda_mem_gb()
                    likely_fragmentation = False
                    if mem:
                        alloc_gb, reserved_gb, free_gb = mem
                        # reserved明显大于alloc，且free也不太小：更可能是缓存/碎片导致分配失败
                        if (reserved_gb - alloc_gb) >= 0.8 and free_gb >= 0.5:
                            likely_fragmentation = True

                    # auto-tune：优先回退 token 预算（不改文本，只减少每次forward的token总量峰值）
                    if auto_tune and token_budget > 0 and lengths_for_debug is not None:
                        current_used = sum(int(lengths_for_debug[j]) for j in batch_indices)
                        new_budget = int(token_budget * (1.0 - token_budget_down_pct))
                        new_budget = max(token_budget_min, min(token_budget_max, new_budget))
                        # 若该batch总token远小于token_budget_min，回退预算并不会改变batch组成，直接走batch_size回退
                        if current_used >= token_budget_min and new_budget < token_budget:
                            stable_batches = 0
                            stable_batches_for_cap = 0
                            oom_cooldown_left = max(oom_cooldown_left, tune_cooldown_batches)
                            token_budget = new_budget
                            success_count = 0
                            oom_count_at_min = 0
                            logger.warning(f"[embed-tune] OOM -> decrease token_budget to {token_budget}, retry...")
                            continue

                    if likely_fragmentation:
                        try:
                            logger.info(f"尝试用当前batch_size={current_batch_size}重新处理...")
                            with torch.inference_mode():
                                batch_result = self.embedding_model.encode(**params)

                            if isinstance(batch_result, torch.Tensor):
                                batch_result = batch_result.detach().to("cpu")
                                if batch_result.dtype == torch.bfloat16:
                                    batch_result = batch_result.to(torch.float32)
                                batch_result = batch_result.numpy()

                            for local_i, original_i in enumerate(batch_indices):
                                outputs[original_i] = batch_result[local_i]
                                processed_indices.append(original_i)

                            # 成功处理，重置OOM计数器
                            oom_count_at_min = 0

                            # 更新进度
                            pos += actual_batch_size
                            success_count += actual_batch_size
                            pbar.update(actual_batch_size)

                            # 每成功处理50条，增加batch size
                            if success_count >= increase_threshold:
                                current_batch_size = min(bs_cap, current_batch_size + 1)
                                success_count = 0
                                logger.info(f"✓ 成功处理50条记录，增加batch_size到 {current_batch_size}")

                            logger.info(f"✓ 清理碎片后成功处理，继续使用batch_size={current_batch_size}")
                            continue
                        except RuntimeError as retry_e:
                            if "out of memory" not in str(retry_e).lower() and "cuda" not in str(retry_e).lower():
                                pbar.close()
                                raise
                            logger.warning(f"清理碎片后仍然OOM，开始降低batch_size...")
                    else:
                        logger.warning("清理后仍可能真实超限，跳过同batch_size重试，直接降低batch_size...")

                    # 第三步：降低batch size
                    if current_batch_size == min_batch_size:
                        # 已经是最小batch size，进行深度清理
                        oom_count_at_min += 1
                        logger.warning(f"✗ batch_size={min_batch_size} OOM (第{oom_count_at_min}/{max_oom_retries}次)")

                        if oom_count_at_min >= max_oom_retries:
                            # 连续5次OOM，返回已编码的部分数据
                            logger.warning(f"⚠ batch_size={min_batch_size}连续{max_oom_retries}次OOM，返回已编码数据")
                            pbar.close()

                            # 合并已编码的结果
                            partial_results = None
                            if processed_indices:
                                # 只返回已成功编码的条目，且按原输入顺序对齐
                                unique = set()
                                ordered = []
                                for idx in processed_indices:
                                    if idx not in unique:
                                        unique.add(idx)
                                        ordered.append(idx)
                                ordered.sort()

                                partial_results = np.stack([outputs[i] for i in ordered if outputs[i] is not None], axis=0)
                                if self.embedding_config.norm:
                                    denom = np.linalg.norm(partial_results, axis=1, keepdims=True)
                                    denom[denom == 0] = 1.0
                                    partial_results = partial_results / denom

                                logger.info(f"✓ 返回已编码的 {len(partial_results)} 条记录")

                            # 创建异常并附加部分结果
                            exc = RuntimeError(f"batch_size={min_batch_size}连续{max_oom_retries}次OOM，需要重新执行")
                            exc.partial_results = partial_results
                            exc.partial_indices = ordered if processed_indices else []
                            exc.processed_count = len(ordered) if processed_indices else 0
                            raise exc

                        else:
                            # 清理GPU缓存并等待
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                            # 根据OOM次数增加等待时间
                            wait_time = oom_count_at_min * 2  # 2秒、4秒、6秒...
                            logger.info(f"等待{wait_time}秒后重试...")
                            time.sleep(wait_time)
                            # 不增加i，重试当前位置
                    else:
                        # 降低batch size（改为除以2）
                        old_batch_size = current_batch_size
                        current_batch_size = max(min_batch_size, current_batch_size // 2)
                        success_count = 0  # 重置计数器
                        oom_count_at_min = 0  # 重置OOM计数器（因为不是在min_batch_size）
                        stable_batches = 0
                        stable_batches_for_cap = 0
                        oom_cooldown_left = max(oom_cooldown_left, tune_cooldown_batches)
                        if auto_tune:
                            # 不要把cap直接砍半到 current_batch_size（会导致后续chunk永久跑得很慢）
                            # 经验上：本次OOM通常发生在 old_batch_size，下一档(old-1)更可能稳定且仍保持吞吐
                            safe_cap = max(min_batch_size, int(old_batch_size) - 1)
                            bs_cap = min(bs_cap, safe_cap)
                            # 下一次（包括本chunk后续/下个chunk）优先从 safe_cap 起跑，避免从过小值爬升浪费时间
                            self._adaptive_state["start_bs"] = int(bs_cap)
                        if speed_tune:
                            speed_test_state = {}
                        logger.warning(f"✗ 检测到OOM，降低batch_size: {old_batch_size} → {current_batch_size}")

                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # 不增加i，重试当前位置
                else:
                    # 其他错误，直接抛出
                    pbar.close()
                    raise

        pbar.close()

        # 合并所有结果
        if any(x is not None for x in outputs):
            if any(x is None for x in outputs):
                raise RuntimeError("部分文本未编码完成，可能发生了未捕获的异常")

            # Persist tuning state for next call (same process).
            self._adaptive_state["bs_cap"] = int(bs_cap)
            # 优先使用已经学到的start_bs（例如OOM后设置的safe_cap）；否则用当前batch_size
            next_start = self._adaptive_state.get("start_bs", None)
            if next_start is None:
                next_start = int(current_batch_size)
            self._adaptive_state["start_bs"] = int(max(min_batch_size, min(bs_cap, int(next_start))))
            if token_budget > 0:
                self._adaptive_state["token_budget"] = int(token_budget)
            if speed_tune:
                self._adaptive_state["speed_test_state"] = speed_test_state

            results = np.stack([x for x in outputs if x is not None], axis=0)
            if self.embedding_config.norm:
                denom = np.linalg.norm(results, axis=1, keepdims=True)
                denom[denom == 0] = 1.0
                results = results / denom

            logger.info(f"✓ 编码完成，共处理 {len(texts)} 条记录")
            return results

        raise RuntimeError("没有成功编码任何数据")
