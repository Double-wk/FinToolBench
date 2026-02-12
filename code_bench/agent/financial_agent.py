from typing import List, Dict, Any, Optional
import json
import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime

# 路径配置
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 假设这些模块存在（按你原有结构）
from code_bench.tools.tools import ToolLoader
from code_bench.utils.model_requests import boyue_model_requests
from code_bench.agent.financial_attribute_analyzer import FinancialAttributeAnalyzer
from code_bench.utils.embedding_retriever import EmbeddingRetriever, LocalCachedEmbedding



logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================
@dataclass
class ToolCall:
    step: int
    tool_name: str
    parameters: Dict[str, Any]
    output: str


@dataclass
class ToolBenchSample:
    id: str
    source_dataset: str
    category: str
    question: str
    response: Optional[List[str]]
    tool_calls: List[ToolCall]
    error: str
    execution_result: str
    ground_truth: str
    setting: Optional[str] = None
    candidate_tools: Optional[List[str]] = None
    select_tools: Optional[List[str]] = None
    


# ==================== Base Agent ====================
class FinancialAgent:
    """基础金融Agent"""

    def __init__(self, tool_path: str = "./tools", embedding_cache_dir="./cache", execution_model_name: str = "ep-20251113093937-p4rll", extract_model_name: str = "ep-20251101221159-hhmrg"):
        # 加载工具
        self.tool_loader = ToolLoader(tool_path)
        self.tool_loader.load_financial_attributes_tools()
        self.tools = self.tool_loader.tools_registry
        self.execution_model_name = execution_model_name
        self.extract_model_name = extract_model_name
        logger.info(f"🔧 已加载 {len(self.tools)} 个工具")
        try:
            self.embedding = LocalCachedEmbedding(
                model_name="bge-m3",
                model_cache_dir="./models"
            )
        except Exception as e:
            logger.warning(f"⚠️ Embedding初始化失败: {e}")
            self.embedding = None

        if self.embedding:
            self.retriever = EmbeddingRetriever(
                self.tools,
                self.embedding,
                cache_dir=embedding_cache_dir
            )
        else:
            self.retriever = None
            logger.warning("⚠️ Embedding不可用，将使用全部工具")

        self.analyzer = FinancialAttributeAnalyzer(self.tools)



    def _execute_with_tools(
        self,
        question: str,
        available_tools: List[Dict],
        max_steps: int = 5,
        inject_attributes: bool = False,
    ) -> Dict:
        """执行工具调用（真实 Function Calling）"""
        system_hint = ""
        if inject_attributes:
            system_hint = "你可以看到工具描述中的 financial_tags 字段。在选择工具时，请参考这些属性来辅助判断：工具的数据更新频率、监管领域以及合规风险。在满足问题需求的前提下，优先选择数据时效性合适，且领域与意图更匹配的工具。"

        messages = []
        if system_hint:
            messages.append({"role": "system", "content": system_hint})
        messages.append({"role": "user", "content": question})
        execution_trace = []
        response_lt = []
        for step in range(max_steps):
            logger.info(f"\n{'='*60}\n🔄 步骤 {step + 1}/{max_steps}\n{'='*60}")

            response = boyue_model_requests(model=self.execution_model_name, messages=messages, available_tools=available_tools)
            if not response:
                logger.error("模型调用失败")
                break

            messages.append(response)
            logger.info(f"{response}")
            tool_calls = getattr(response, 'tool_calls', None)
            
            response_lt.append(response.reasoning_content)
            logger.info(f"{response.reasoning_content}")
            if not tool_calls:
                logger.info("没有工具调用，进入总结阶段")

                final_answer = (
                    response.content.strip()
                    if response and response.content and response.content.strip()
                    else None
                )

                if not final_answer:
                    logger.warning("模型返回空内容，使用纯生成模型兜底一次")
                    retry_messages = messages + [
                        {
                            "role": "system",
                            "content": "请直接给出最终答案，不要调用工具。"
                        }
                    ]
                    final_resp = boyue_model_requests(
                        model=self.execution_model_name,
                        messages=retry_messages,
                        available_tools=[]
                    )
                    final_answer = (
                        final_resp.content.strip()
                        if final_resp and final_resp.content and final_resp.content.strip()
                        else "模型未生成有效回答"
                    )
  
                return {
                    "response": response_lt,
                    "tool_calls": execution_trace,
                    "final_answer": final_answer,
                    "error": None
                }

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    func_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    func_args = {}

                logger.info(f"\n🔧 调用: {func_name}\n   参数: {json.dumps(func_args, ensure_ascii=False)}")

                try:
                    tool_func = self.tool_loader.get_tool_function(func_name)
                    observation = tool_func(**func_args)
                    if hasattr(observation, 'to_dict'):
                        observation = observation.to_dict(orient="records")
                    observation_str = json.dumps(observation, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"工具执行出错: {e}")
                    observation_str = json.dumps({"error": str(e)}, ensure_ascii=False)
                
                
                if len(observation_str) >= 500:
                    logger.info("  📌 输出过长，正在提取相关信息...")
                    try:
                        # observation_str_original = observation_str
                        # 构建摘要 prompt
                        summary_prompt = (
                            f"你是一个金融助手。用户的问题是：\n{question}\n\n"
                            f"工具返回了以下数据（可能很长）：\n{observation_str}\n\n"
                            "请从中提取**最直接回答用户问题的关键信息**，保持简洁，不要添加解释或格式。"
                        )
                        summary_resp = boyue_model_requests(model=self.extract_model_name,
                            messages=[{"role": "user", "content": summary_prompt}]
                        )
                        if summary_resp:
                            logger.info(f"  ✅ 摘要成功，长度从 {len(observation_str)} 压缩到 {len(summary_resp)}")
                            observation_str = summary_resp.strip()
                        else:
                            observation_str = "工具返回内容过长"
                            logger.warning("  ⚠️ 摘要失败，保留原始输出")
                    except Exception as summarize_error:
                        observation_str = "工具返回内容过长"
                        logger.error(f"  ❌ 摘要过程出错: {summarize_error}")
                
                
                execution_trace.append({
                    "step": step + 1,
                    "tool_name": func_name,
                    "parameters": func_args,
                    "output": observation_str
                })

                logger.info(f"📥 返回: {observation_str[:200]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": observation_str
                })

        error = None
        if not execution_trace:
            logger.error("execution_trace is empty (LLM may have failed before any tool call).")
            error = "execution_trace_empty"
        else:
            if 'error' in execution_trace[-1]['output']:
                error = execution_trace[-1]['output']
        # 最终总结
        try:
            final_resp = boyue_model_requests(model=self.execution_model_name, messages=messages, available_tools=[])
            final_answer = final_resp.content if final_resp else "任务未完成"
        except Exception as e:
            error = e
            final_answer = f"最终总结失败: {e}"

        return {
            "response": response_lt,
            "tool_calls": execution_trace,
            "final_answer": final_answer,
            "error": error
        }
        
    def _create_sample(self, idx, per_data, result, setting: Optional[str], candidate_tools: List[str]) -> ToolBenchSample:
        return ToolBenchSample(
            id=f"{per_data['dataset_name']}_{idx}",
            source_dataset=per_data['dataset_name'],
            category=per_data['category'], 
            question=per_data['question'],
            tool_calls=result['tool_calls'],
            response=result['response'],
            execution_result=result['final_answer'],
            error=result['error'],
            ground_truth=str(per_data['answer']),
            setting=setting,
            candidate_tools=candidate_tools,
        )
    
    def run(
        self,
        idx,
        per_data: str,
        top_k: int = 20,
        max_steps: int = 5,
        inject_attributes: bool = False,
        setting: Optional[str] = None
    ) -> Dict:
        question = per_data['question']
        logger.info(f"\n{'='*100}")
        logger.info("🚀 agent调用（Function Calling 版）")
        logger.info(f"{'='*100}")
        logger.info(f"问题: {question}")

        # ===== 阶段1: 检索候选工具 =====
        if self.retriever:
            candidate_tool_names = self.retriever.retrieve_top_k(question, top_k=top_k)
        else:
            # return None
            candidate_tool_names = list(self.tools.keys())[:top_k]
            logger.warning("⚠️ 使用全部工具（Embedding不可用）")

        logger.info(f"✅ 检索到 {len(candidate_tool_names)} 个候选工具")

        logger.info("加载带金融属性的schemas")
        # 构建两种工具 schema
        financial_schemas = []

        for name in candidate_tool_names:
            tool = self.tools[name]
            enhanced_desc = f"{tool['schema']['function']['description']}"
            if inject_attributes:
                attrs = self.analyzer.get_attributes(name)
                if attrs:
                    enhanced_desc = (
                        f"{enhanced_desc}\nfinancial_tags: {json.dumps(attrs, ensure_ascii=False)}"
                    )
            fin_schema = {
                "type": "function",
                "function": {
                    "name": tool["schema"]["function"]["name"],
                    "description": enhanced_desc,
                    "parameters": tool["schema"]["function"]["parameters"]
                }
            }
            financial_schemas.append(fin_schema)

        # ===== 执行agent =====
        logger.info("\n🔵 执行agent...")
        result = self._execute_with_tools(
            question, financial_schemas, max_steps, inject_attributes=inject_attributes
        )
        
        return self._create_sample(idx, per_data, result, setting, candidate_tool_names)
