import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict, Counter
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict, Counter
import re
import time

from .llm import _get_llm_class, BaseLLM, _find_existing_working_dir
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class StandardRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 embedding_model_name=None,
                 llm_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        # 使用模糊匹配查找已存在的工作目录（LLM模糊 + Embedding精确）
        self.working_dir = _find_existing_working_dir(
            self.global_config.save_dir,
            self.global_config.llm_name,
            self.global_config.embedding_model_name
        )

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        # StandardRAG使用独立的缓存目录
        self.llm_model: BaseLLM = _get_llm_class(self.global_config, method='standardrag')

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)

        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self.last_retrieval_mode_counts: Dict[str, int] = {}

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"索引文档")

        self.chunk_embedding_store.insert_strings(docs)

    def delete(self, docs_to_delete: List[str]):
        """

        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        logger.info(f"删除 {len(chunk_ids_to_delete)} 个文档块")

        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        self.ready_to_retrieve = False

    def retrieve(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []
        mode_counts = Counter()

        # 准备保存retrieval responses到JSONL文件
        retrieval_dump_path = os.path.join(self.working_dir, "retrieval_responses.jsonl")
        retrieval_dump_file = None

        if retrieval_dump_path:
            dump_dir = os.path.dirname(retrieval_dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)

            # 检查文件是否已存在
            file_exists = os.path.exists(retrieval_dump_path)

            # 去重：检查最后一次运行的Retrieval记录是否与当前相同
            should_skip = False
            if file_exists:
                try:
                    with open(retrieval_dump_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # 找到最后一个run_separator和对应的Retrieval记录
                    last_retrieval_records = []
                    last_separator = None
                    for i in range(len(lines) - 1, -1, -1):
                        line = lines[i].strip()
                        if not line or line.startswith("="):
                            continue
                        try:
                            record = json.loads(line)
                            if record.get("type") == "run_separator":
                                last_separator = record
                                break
                            if "question" in record:  # 这是Retrieval记录
                                last_retrieval_records.insert(0, record)
                        except:
                            continue

                    # 只有当最后一次运行的method与当前相同且模型相同时才检查去重
                    if last_separator and last_separator.get("method") == "StandardRAG":
                        # 检查模型是否相同（使用模糊匹配）
                        from .llm import _extract_model_keywords
                        last_llm_keywords = _extract_model_keywords(last_separator.get("llm_name", ""))
                        current_llm_keywords = _extract_model_keywords(self.global_config.llm_name)
                        models_match = (last_llm_keywords == current_llm_keywords) if last_llm_keywords and current_llm_keywords else (last_separator.get("llm_name") == self.global_config.llm_name)

                        if models_match and len(last_retrieval_records) == len(queries):
                            all_same = True
                            for idx, query in enumerate(queries):
                                last_record = last_retrieval_records[idx]
                                if last_record.get("question") != query:
                                    all_same = False
                                    break

                            if all_same:
                                should_skip = True
                                logger.info("检索响应与上次运行相同，跳过保存以避免重复")
                except Exception as e:
                    logger.debug(f"Failed to check for duplicate Retrieval records: {e}")

            if should_skip:
                retrieval_dump_file = None
            else:
                retrieval_dump_file = open(retrieval_dump_path, "a", encoding="utf-8")

                # 如果文件已存在，添加分隔行
                if file_exists:
                    retrieval_dump_file.write("\n")
                    retrieval_dump_file.write("\n")
                    retrieval_dump_file.write("\n")

                # 写入本次运行的时间戳和配置信息
                from datetime import datetime
                run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                separator_record = {
                    "type": "run_separator",
                    "timestamp": run_timestamp,
                    "method": "StandardRAG",
                    "llm_name": self.global_config.llm_name,
                    "llm_base_url": self.global_config.llm_base_url,
                    "embedding_name": self.global_config.embedding_model_name,
                    "dataset": self.global_config.dataset,
                    "num_queries": len(queries),
                }
                retrieval_dump_file.write(json.dumps(separator_record, ensure_ascii=False, default=str) + "\n")
                retrieval_dump_file.write("=" * 80 + "\n")

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            logger.info(f"[Retrieval][DPR] Query #{q_idx}: dense retrieval (StandardRAG).")
            mode_counts['dpr'] += 1

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

            # 保存retrieval记录到JSONL
            if retrieval_dump_file:
                from datetime import datetime
                retrieval_dump_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # 精确到毫秒
                    "index": q_idx,
                    "question": query,
                    "retrieval_mode": "dpr",
                    "num_retrieved_docs": len(top_k_docs),
                    "top_doc_scores": [float(score) for score in sorted_doc_scores[:5]],  # 只保存前5个分数
                    "retrieved_doc_previews": [doc[:100] + "..." if len(doc) > 100 else doc for doc in top_k_docs[:3]],  # 只保存前3个文档的预览
                }
                retrieval_dump_file.write(json.dumps(retrieval_dump_record, ensure_ascii=False, default=str) + "\n")

        # 关闭retrieval dump文件
        if retrieval_dump_file:
            retrieval_dump_file.close()
            logger.info("检索响应已保存到 %s", retrieval_dump_path)

        retrieve_end_time = time.time()  # Record end time
        self.last_retrieval_mode_counts = dict(mode_counts)

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"总检索时间 {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            overall_retrieval_result["retrieval_mode_counts"] = dict(mode_counts)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            # Save retrieval metrics to file (only if not called from rag_qa)
            # rag_qa will save both retrieval and qa metrics together
            if not hasattr(self, '_skip_retrieval_save') or not self._skip_retrieval_save:
                self._save_evaluation_metrics(overall_retrieval_result, None, source="StandardRAG.retrieve")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def _save_evaluation_metrics(self,
                                 retrieval_metrics: Optional[Dict] = None,
                                 qa_metrics: Optional[Dict] = None,
                                 source: Optional[str] = None):
        """
        Save evaluation metrics to file by appending a new run entry with a timestamp.
        This preserves previous runs rather than overwriting them.

        Args:
            retrieval_metrics: Dictionary containing retrieval evaluation metrics (e.g., Recall@k)
            qa_metrics: Dictionary containing QA evaluation metrics (e.g., ExactMatch, F1)
        """
        metrics_file_path = os.path.join(self.working_dir, "evaluation_metrics.json")

        if retrieval_metrics is None and qa_metrics is None:
            return

        # Load existing metrics if file exists
        existing_payload = {}
        runs = []
        if os.path.exists(metrics_file_path):
            try:
                with open(metrics_file_path, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load existing metrics file: {e}. Starting with empty metrics.")
                existing_data = None

            if isinstance(existing_data, dict):
                if isinstance(existing_data.get("runs"), list):
                    runs = existing_data.get("runs", [])
                    existing_payload = {k: v for k, v in existing_data.items() if k != "runs"}
                else:
                    existing_payload = {"legacy_metrics": existing_data}
            elif isinstance(existing_data, list):
                runs = existing_data

        timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
        run_entry = {
            "timestamp": timestamp,
            "llm_name": getattr(self.global_config, "llm_name", None),
            "embedding_model_name": getattr(self.global_config, "embedding_model_name", None),
            "dataset": getattr(self.global_config, "dataset", None),
            "source": source or self.__class__.__name__,
        }
        if retrieval_metrics is not None and getattr(self, "last_retrieval_mode_counts", None):
            run_entry["retrieval_mode_counts"] = self.last_retrieval_mode_counts
        if retrieval_metrics is not None:
            run_entry["retrieval_metrics"] = retrieval_metrics
        if qa_metrics is not None:
            run_entry["qa_metrics"] = qa_metrics

        # 去重：检查是否与最后一次运行的结果相同（且source和模型相同）
        if runs:
            last_run = runs[-1]
            # 检查source是否相同
            if last_run.get("source") == run_entry.get("source"):
                # 检查模型是否相同（使用模糊匹配）
                from .llm import _extract_model_keywords
                last_llm_keywords = _extract_model_keywords(last_run.get("llm_name", ""))
                current_llm_keywords = _extract_model_keywords(run_entry.get("llm_name", ""))
                models_match = (last_llm_keywords == current_llm_keywords) if last_llm_keywords and current_llm_keywords else (last_run.get("llm_name") == run_entry.get("llm_name"))

                if models_match:
                    # 比较metrics内容（忽略timestamp）
                    is_duplicate = True
                    for key in ["retrieval_metrics", "qa_metrics", "retrieval_mode_counts"]:
                        if run_entry.get(key) != last_run.get(key):
                            is_duplicate = False
                            break

                    if is_duplicate:
                        logger.info(f"指标与上次运行相同，跳过保存以避免重复")
                        return

        runs.append(run_entry)
        evaluation_payload = dict(existing_payload)
        evaluation_payload["runs"] = runs

        # 自定义JSON格式化：在retrieval和qa之间加小间隔，不同运行之间加大间隔
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            f.write("{\n")

            # 写入其他字段（如果有）
            other_keys = [k for k in evaluation_payload.keys() if k != "runs"]
            for i, key in enumerate(other_keys):
                f.write(f'  "{key}": {json.dumps(evaluation_payload[key], ensure_ascii=False)}')
                if i < len(other_keys) - 1 or "runs" in evaluation_payload:
                    f.write(",")
                f.write("\n")

            # 写入runs数组
            f.write('  "runs": [\n')
            for run_idx, run in enumerate(runs):
                f.write("    {\n")

                # 写入基本信息
                basic_keys = ["timestamp", "llm_name", "embedding_model_name", "dataset", "source"]
                for key in basic_keys:
                    if key in run:
                        f.write(f'      "{key}": {json.dumps(run[key], ensure_ascii=False)},\n')

                # 写入retrieval_mode_counts（如果有）
                if "retrieval_mode_counts" in run:
                    f.write(f'      "retrieval_mode_counts": {json.dumps(run["retrieval_mode_counts"], ensure_ascii=False)},\n')

                # 写入retrieval_metrics（如果有）
                if "retrieval_metrics" in run:
                    f.write('      "retrieval_metrics": ')
                    f.write(json.dumps(run["retrieval_metrics"], ensure_ascii=False))
                    if "qa_metrics" in run:
                        f.write(',\n\n')  # 小间隔：retrieval和qa之间
                    else:
                        f.write('\n')

                # 写入qa_metrics（如果有）
                if "qa_metrics" in run:
                    f.write('      "qa_metrics": ')
                    f.write(json.dumps(run["qa_metrics"], ensure_ascii=False))
                    f.write('\n')

                # 结束当前run
                f.write("    }")
                if run_idx < len(runs) - 1:
                    f.write(",\n\n\n")  # 大间隔：不同运行之间（3个换行）
                else:
                    f.write("\n")

            f.write("  ]\n")
            f.write("}\n")

        logger.info(f"评估指标已保存到 {metrics_file_path}")

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            # 设置标志，让retrieve()不保存metrics（我们会在这里一次性保存）
            self._skip_retrieval_save = True
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)
            self._skip_retrieval_save = False

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            # Save both retrieval and QA metrics together in one record
            self._save_evaluation_metrics(overall_retrieval_result, overall_qa_results, source="StandardRAG.rag_qa")

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # 为QA阶段创建独立的LLM实例，使用qa_cache.sqlite
        qa_llm_model = _get_llm_class(self.global_config, cache_suffix='qa', method='standardrag')
        logger.info(f"QA LLM使用独立缓存: qa_cache.sqlite (StandardRAG)")

        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        qa_num_workers = max(1, int(getattr(self.global_config, "qa_num_workers", 1) or 1))
        if qa_num_workers == 1:
            all_qa_results = [
                qa_llm_model.infer(qa_messages)
                for qa_messages in tqdm(all_qa_messages, desc="QA Reading")
            ]
        else:
            with ThreadPoolExecutor(max_workers=qa_num_workers) as executor:
                results_iterator = executor.map(qa_llm_model.infer, all_qa_messages)
                all_qa_results = list(
                    tqdm(results_iterator, total=len(all_qa_messages), desc="QA Reading")
                )

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message = list(all_response_message)
        all_metadata = list(all_metadata)
        all_cache_hit = list(all_cache_hit)

        # 自动设置QA导出路径（如果用户未指定）
        qa_dump_path = getattr(self.global_config, "qa_dump_path", None)
        if qa_dump_path is None:
            # 自动生成路径：outputs/<dataset>/<llm>_<embedding>/qa_responses.jsonl
            qa_dump_path = os.path.join(self.working_dir, "qa_responses.jsonl")
            logger.info(f"QA导出路径未指定，自动生成: {qa_dump_path}")

        qa_dump_file = None
        if qa_dump_path:
            dump_dir = os.path.dirname(qa_dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)

            # 检查文件是否已存在，如果存在则追加分隔符
            file_exists = os.path.exists(qa_dump_path)

            # 去重：检查最后一次运行的QA记录是否与当前相同
            should_skip = False
            if file_exists:
                try:
                    with open(qa_dump_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # 找到最后一个run_separator和对应的QA记录
                    last_qa_records = []
                    last_separator = None
                    for i in range(len(lines) - 1, -1, -1):
                        line = lines[i].strip()
                        if not line or line.startswith("="):
                            continue
                        try:
                            record = json.loads(line)
                            if record.get("type") == "run_separator":
                                last_separator = record
                                break
                            if "question" in record:  # 这是QA记录
                                last_qa_records.insert(0, record)
                        except:
                            continue

                    # 只有当最后一次运行的method与当前相同且模型相同时才检查去重
                    if last_separator and last_separator.get("method") == "StandardRAG":
                        # 检查模型是否相同（使用模糊匹配）
                        from .llm import _extract_model_keywords
                        last_llm_keywords = _extract_model_keywords(last_separator.get("llm_name", ""))
                        current_llm_keywords = _extract_model_keywords(self.global_config.llm_name)
                        models_match = (last_llm_keywords == current_llm_keywords) if last_llm_keywords and current_llm_keywords else (last_separator.get("llm_name") == self.global_config.llm_name)

                        if models_match and len(last_qa_records) == len(queries):
                            all_same = True
                            for idx, query in enumerate(queries):
                                last_record = last_qa_records[idx]
                                current_response = all_response_message[idx]
                                if last_record.get("question") != query.question or last_record.get("response") != current_response:
                                    all_same = False
                                    break

                            if all_same:
                                should_skip = True
                                logger.info("问答响应与上次运行相同，跳过保存以避免重复")
                except Exception as e:
                    logger.debug(f"Failed to check for duplicate QA records: {e}")

            if should_skip:
                qa_dump_file = None
            else:
                qa_dump_file = open(qa_dump_path, "a", encoding="utf-8")  # 使用追加模式

                # 如果文件已存在，添加分隔行
                if file_exists:
                    qa_dump_file.write("\n")  # 空行
                    qa_dump_file.write("\n")  # 空行
                    qa_dump_file.write("\n")  # 空行

                # 写入本次运行的时间戳和配置信息
                from datetime import datetime
                run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                separator_record = {
                    "type": "run_separator",
                    "timestamp": run_timestamp,
                    "method": "StandardRAG",
                    "llm_name": self.global_config.llm_name,
                    "llm_base_url": self.global_config.llm_base_url,
                    "embedding_name": self.global_config.embedding_model_name,
                    "dataset": self.global_config.dataset,
                    "num_queries": len(queries),
                }
                qa_dump_file.write(json.dumps(separator_record, ensure_ascii=False, default=str) + "\n")
                qa_dump_file.write("=" * 80 + "\n")  # 分隔线

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            metadata = all_metadata[query_solution_idx] if query_solution_idx < len(all_metadata) else {}

            invalid_reason = None
            if response_content is None:
                invalid_reason = "empty_response"
            elif not isinstance(response_content, str):
                invalid_reason = f"non_string_response:{type(response_content).__name__}"
            else:
                stripped_response = response_content.strip()
                if not stripped_response:
                    invalid_reason = "blank_response"
                else:
                    lower_response = stripped_response.lower()
                    if isinstance(metadata, dict) and metadata.get("finish_reason") == "error":
                        invalid_reason = "finish_reason_error"
                    elif stripped_response.startswith("{") and "error" in lower_response:
                        invalid_reason = "error_json"
                    elif lower_response.startswith(("error", "exception", "traceback")):
                        invalid_reason = "error_prefix"
                    elif "rate limit" in lower_response or "quota" in lower_response:
                        invalid_reason = "rate_limit"

            if invalid_reason:
                logger.warning(
                    "Invalid QA response detected (idx=%s, reason=%s).",
                    query_solution_idx,
                    invalid_reason,
                )
                if isinstance(metadata, dict):
                    metadata["qa_error"] = True
                    metadata["qa_error_reason"] = invalid_reason
                pred_ans = ""
            else:
                try:
                    pred_ans = response_content.split('Answer:')[1].strip()
                except Exception as e:
                    logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                    pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

            if qa_dump_file:
                from datetime import datetime
                qa_dump_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # 精确到毫秒
                    "index": query_solution_idx,
                    "question": query_solution.question,
                    "answer": pred_ans,
                    "response": response_content,
                    "metadata": metadata,
                    "cache_hit": all_cache_hit[query_solution_idx] if query_solution_idx < len(all_cache_hit) else None,
                    "invalid_reason": invalid_reason,
                }
                qa_dump_file.write(json.dumps(qa_dump_record, ensure_ascii=False, default=str) + "\n")

        if qa_dump_file:
            qa_dump_file.close()
            logger.info("问答响应已保存到 %s", qa_dump_path)

        return queries_solutions, all_response_message, all_metadata


    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("准备快速检索")

        logger.info("加载键")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys

        logger.info("加载嵌入")
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"为 {len(all_query_strings)} 个查询编码 query_to_passage")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
