import os
import glob

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _extract_model_keywords(model_name: str) -> list:
    """
    从模型名称中提取关键词用于模糊匹配

    Args:
        model_name: 模型名称

    Returns:
        关键词列表
    """
    model_lower = model_name.lower()
    keywords = []

    # Llama系列
    if 'llama' in model_lower:
        keywords.append('llama')
        if '3.3' in model_lower and '70b' in model_lower:
            keywords.extend(['3.3', '70b'])
        elif '3.1' in model_lower and '8b' in model_lower:
            keywords.extend(['3.1', '8b'])

    return keywords


def _fuzzy_match_filename(filename: str, llm_keywords: list, embedding_name: str = None) -> bool:
    """
    模糊匹配文件名

    Args:
        filename: 文件名
        llm_keywords: LLM关键词列表
        embedding_name: Embedding模型名称（可选，如果提供则需要精确匹配）

    Returns:
        是否匹配
    """
    filename_lower = filename.lower()

    # 检查LLM关键词（模糊匹配）
    if not all(keyword in filename_lower for keyword in llm_keywords):
        return False

    # 如果指定了embedding，需要精确匹配
    if embedding_name:
        embedding_normalized = embedding_name.replace('/', '_').replace(':', '_').lower()
        if embedding_normalized not in filename_lower:
            return False

    return True


def _find_existing_openie_file(save_dir: str, llm_name: str) -> str:
    """
    模糊匹配已存在的OpenIE结果文件（只匹配LLM名称）

    Args:
        save_dir: 保存目录
        llm_name: 当前LLM名称

    Returns:
        匹配到的文件路径，如果没有匹配则返回新文件路径
    """
    llm_keywords = _extract_model_keywords(llm_name)
    pattern = os.path.join(save_dir, "openie_results_ner_*.json")
    existing_files = glob.glob(pattern)

    # 模糊匹配
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        if _fuzzy_match_filename(filename, llm_keywords):
            logger.info(f"找到匹配的OpenIE结果文件: {filename}")
            return filepath

    # 没有匹配，返回新文件路径
    llm_label = llm_name.replace('/', '_').replace(':', '_')
    new_filepath = os.path.join(save_dir, f"openie_results_ner_{llm_label}.json")
    logger.info(f"未找到匹配的OpenIE结果文件，将创建: {os.path.basename(new_filepath)}")
    return new_filepath


def _find_existing_working_dir(save_dir: str, llm_name: str, embedding_name: str) -> str:
    """
    模糊匹配已存在的工作目录（LLM模糊匹配 + Embedding精确匹配）

    Args:
        save_dir: 保存目录
        llm_name: 当前LLM名称
        embedding_name: 当前Embedding名称

    Returns:
        匹配到的工作目录路径
    """
    llm_keywords = _extract_model_keywords(llm_name)
    embedding_normalized = embedding_name.replace('/', '_').replace(':', '_')

    # 查找所有可能的目录
    pattern = os.path.join(save_dir, "*")
    existing_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    # 模糊匹配
    for dirpath in existing_dirs:
        dirname = os.path.basename(dirpath)
        if _fuzzy_match_filename(dirname, llm_keywords, embedding_name):
            logger.info(f"找到匹配的工作目录: {dirname}")
            return dirpath

    # 没有匹配，返回新目录路径
    llm_label = llm_name.replace('/', '_').replace(':', '_')
    new_dirpath = os.path.join(save_dir, f"{llm_label}_{embedding_normalized}")
    logger.info(f"未找到匹配的工作目录，将创建: {os.path.basename(new_dirpath)}")
    return new_dirpath


def _find_existing_cache_file(cache_dir: str, cache_suffix: str, llm_name: str, embedding_name: str = None, method: str = 'hipporag') -> str:
    """
    模糊匹配已存在的缓存文件

    Args:
        cache_dir: 缓存根目录
        cache_suffix: 缓存后缀（'openie', 'retrieval', 'qa'）
        llm_name: 当前LLM名称
        embedding_name: 当前Embedding名称（retrieval和qa需要）
        method: 方法名称（'hipporag' 或 'standardrag'）

    Returns:
        缓存文件的完整路径（包含目录）
    """
    llm_keywords = _extract_model_keywords(llm_name)
    llm_label = llm_name.replace('/', '_').replace(':', '_')

    if cache_suffix == 'openie':
        # OpenIE缓存在根目录，只匹配LLM，文件名格式：{llm}_openie_cache_{method}.sqlite
        pattern = os.path.join(cache_dir, f"*_openie_cache_{method}.sqlite")
        existing_files = glob.glob(pattern)

        for filepath in existing_files:
            filename = os.path.basename(filepath)
            if _fuzzy_match_filename(filename, llm_keywords):
                logger.info(f"找到匹配的OpenIE缓存: {filename}")
                return filepath

        # 没有匹配，返回新文件路径
        new_filepath = os.path.join(cache_dir, f"{llm_label}_openie_cache_{method}.sqlite")
        logger.info(f"未找到匹配的OpenIE缓存，将创建: {os.path.basename(new_filepath)}")
        return new_filepath

    else:
        # Retrieval和QA缓存在子目录，需要匹配LLM+Embedding
        if not embedding_name:
            raise ValueError(f"cache_suffix={cache_suffix}需要提供embedding_name参数")

        embedding_normalized = embedding_name.replace('/', '_').replace(':', '_')

        # 查找匹配的子目录
        pattern = os.path.join(cache_dir, "*")
        existing_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

        matched_subdir = None
        for dirpath in existing_dirs:
            dirname = os.path.basename(dirpath)
            if _fuzzy_match_filename(dirname, llm_keywords, embedding_name):
                matched_subdir = dirpath
                logger.info(f"找到匹配的缓存子目录: {dirname}")
                break

        # 如果没有匹配的子目录，创建新的
        if not matched_subdir:
            matched_subdir = os.path.join(cache_dir, f"{llm_label}_{embedding_normalized}")
            logger.info(f"未找到匹配的缓存子目录，将创建: {os.path.basename(matched_subdir)}")

        os.makedirs(matched_subdir, exist_ok=True)

        # 返回子目录中的缓存文件路径，文件名格式：{cache_suffix}_cache_{method}.sqlite
        cache_filepath = os.path.join(matched_subdir, f"{cache_suffix}_cache_{method}.sqlite")
        return cache_filepath


def _get_llm_class(config: BaseConfig, cache_suffix: str = None, method: str = 'hipporag'):
    """
    获取LLM类实例

    Args:
        config: 全局配置
        cache_suffix: 缓存文件后缀，用于区分不同用途的缓存
                     - 'openie': OpenIE阶段（只匹配LLM）
                     - 'retrieval': Retrieval阶段（匹配LLM+Embedding）
                     - 'qa': QA阶段（匹配LLM+Embedding）
                     - None: 默认缓存（向后兼容）
        method: 方法名称，用于区分不同的RAG方法
                     - 'hipporag': HippoRAG方法
                     - 'standardrag': StandardRAG方法
    """
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)

    # 为CacheOpenAI传递cache_suffix参数
    # 所有方法使用同一个llm_cache目录，通过文件名后缀区分
    cache_dir = os.path.join(config.save_dir, "llm_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if cache_suffix:
        # 使用模糊匹配查找已存在的缓存文件，传递method参数
        embedding_name = config.embedding_model_name if cache_suffix in ['retrieval', 'qa'] else None
        cache_filepath = _find_existing_cache_file(cache_dir, cache_suffix, config.llm_name, embedding_name, method)
        return CacheOpenAI(cache_dir=os.path.dirname(cache_filepath),
                          global_config=config,
                          cache_filename=os.path.basename(cache_filepath))
    else:
        # 默认缓存（向后兼容）
        return CacheOpenAI.from_experiment_config(config)
    