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

    def batch_encode(self, texts: List[str], **kwargs) -> None:
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
        current_batch_size = 4  # 初始batch size
        max_batch_size = 12  # 最大batch size
        min_batch_size = 1  # 最小batch size
        success_count = 0  # 成功处理的记录数
        increase_threshold = 100  # 每处理100条增加batch size
        oom_count_at_min = 0  # 在batch_size=1时连续OOM的次数
        max_oom_retries = 5  # 最大连续OOM重试次数

        logger.info(f"开始动态batch size编码，初始batch_size={current_batch_size}，最大={max_batch_size}")

        pbar = tqdm(total=len(texts), desc="Batch Encoding")
        results = []
        i = 0

        while i < len(texts):
            # 确定当前批次大小
            actual_batch_size = min(current_batch_size, len(texts) - i)
            params["prompts"] = texts[i:i + actual_batch_size]

            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 编码当前批次
                batch_result = self.embedding_model.encode(**params)
                results.append(batch_result)

                # 成功处理，重置OOM计数器
                oom_count_at_min = 0

                # 更新进度
                i += actual_batch_size
                success_count += actual_batch_size
                pbar.update(actual_batch_size)

                # 每成功处理100条，增加batch size
                if success_count >= increase_threshold and current_batch_size < max_batch_size:
                    current_batch_size += 1
                    success_count = 0  # 重置计数器
                    logger.info(f"✓ 成功处理100条记录，增加batch_size到 {current_batch_size}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    # OOM错误处理：先清理碎片，再降低batch size
                    logger.warning(f"✗ 检测到OOM，开始清理GPU碎片...")

                    # 第一步：清理GPU碎片
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # 同步GPU操作

                    # 第二步：尝试用当前batch size重试（清理碎片后可能成功）
                    try:
                        logger.info(f"尝试用当前batch_size={current_batch_size}重新处理...")
                        batch_result = self.embedding_model.encode(**params)
                        results.append(batch_result)

                        # 成功处理，重置OOM计数器
                        oom_count_at_min = 0

                        # 更新进度
                        i += actual_batch_size
                        success_count += actual_batch_size
                        pbar.update(actual_batch_size)

                        # 每成功处理100条，增加batch size
                        if success_count >= increase_threshold and current_batch_size < max_batch_size:
                            current_batch_size += 1
                            success_count = 0
                            logger.info(f"✓ 成功处理100条记录，增加batch_size到 {current_batch_size}")

                        logger.info(f"✓ 清理碎片后成功处理，继续使用batch_size={current_batch_size}")
                        continue

                    except RuntimeError as retry_e:
                        if "out of memory" not in str(retry_e).lower() and "cuda" not in str(retry_e).lower():
                            pbar.close()
                            raise

                        # 清理碎片后仍然OOM，降低batch size
                        logger.warning(f"清理碎片后仍然OOM，开始降低batch_size...")

                    # 第三步：降低batch size
                    if current_batch_size == min_batch_size:
                        # 已经是最小batch size，进行深度清理
                        oom_count_at_min += 1
                        logger.warning(f"✗ batch_size={min_batch_size} OOM (第{oom_count_at_min}/{max_oom_retries}次)")

                        if oom_count_at_min >= max_oom_retries:
                            # 连续5次OOM，进行最后的深度清理尝试
                            logger.warning(f"⚠ 连续{max_oom_retries}次OOM，执行深度清理...")

                            # 深度清理GPU
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                torch.cuda.ipc_collect()  # 清理IPC缓存

                            # 等待GPU完全释放
                            logger.info("等待5秒让GPU完全释放...")
                            time.sleep(5)

                            # 重置OOM计数器，再给5次机会
                            oom_count_at_min = 0
                            logger.info(f"✓ 深度清理完成，重置计数器，继续尝试...")
                            # 不增加i，重试当前位置
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
                        # 降低batch size（每次减1而不是减半）
                        old_batch_size = current_batch_size
                        current_batch_size = max(min_batch_size, current_batch_size - 1)
                        success_count = 0  # 重置计数器
                        oom_count_at_min = 0  # 重置OOM计数器（因为不是在min_batch_size）
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
        if results:
            results = torch.cat(results, dim=0)

            if isinstance(results, torch.Tensor):
                results = results.cpu()
                results = results.numpy()
            if self.embedding_config.norm:
                results = (results.T / np.linalg.norm(results, axis=1)).T

            logger.info(f"✓ 编码完成，共处理 {len(texts)} 条记录")
            return results
        else:
            raise RuntimeError("没有成功编码任何数据")
