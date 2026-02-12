"""
Financial Agent with Two-Stage Tool Selection + Embedding Cache
两阶段工具选择：Embedding检索 + 金融属性对比 + 缓存优化
"""
import os
import re
import sys
import json
import pickle
import logging
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 路径配置
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code_bench.tools.tools import ToolLoader

# ==================== Embedding 配置 ====================
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logging.warning("⚠️ sentence-transformers未安装")
    logging.warning("   安装命令: pip install -U sentence-transformers")


logger = logging.getLogger(__name__)


# ==================== Local Cached Embedding ====================
class LocalCachedEmbedding:
    """本地缓存Embedding模型"""
    
    RECOMMENDED_MODELS = {
        "bge-small": {
            "hf_name": "BAAI/bge-small-zh-v1.5",
            "size": "~100MB",
            "dim": 512,
            "description": "轻量高效，推荐日常使用"
        },
        "bge-base": {
            "hf_name": "BAAI/bge-base-zh-v1.5",
            "size": "~400MB", 
            "dim": 768,
            "description": "质量更好，推荐重要场景"
        },
        "bge-large": {
            "hf_name": "BAAI/bge-large-zh-v1.5",
            "size": "~1.3GB",
            "dim": 1024,
            "description": "最高质量，适合离线批处理"
        },
         "bge-m3": {
            "hf_name": "BAAI/bge-m3",
            "size": "～2.0GB (FP16)",
            "dim": 1024,
            "description": "SOTA 多语言嵌入模型，支持稠密+稀疏混合检索，金融/专业场景首选"
        },
        "minilm-l6": {
            "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
            "size": "～90MB",
            "dim": 384,
            "description": "轻量英文 embedding 模型，速度快，适合多语言基础任务（主英文）"
        }
    }
    
    def __init__(self, model_name: str = "bge-small", model_cache_dir: str = "./models", device: str = None):
        if not EMBEDDING_AVAILABLE:
            raise ImportError("请安装: pip install -U sentence-transformers")

        self.model_cache_dir = Path(model_cache_dir).absolute()
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name in self.RECOMMENDED_MODELS:
            self.model_info = self.RECOMMENDED_MODELS[model_name]
            hf_model_name = self.model_info["hf_name"]
            logger.info(f"📦 使用推荐模型: {model_name}")
            logger.info(f"   HuggingFace: {hf_model_name}")
            logger.info(f"   {self.model_info['description']}")
            logger.info(f"   大小: {self.model_info['size']}, 维度: {self.model_info['dim']}")
        else:
            hf_model_name = model_name
            self.model_info = {"hf_name": model_name}
            logger.info(f"📦 使用自定义模型: {model_name}")
        

        logger.info(f"📁 模型缓存目录: {self.model_cache_dir}")
        try:
            logger.info(f"🔄 加载模型中...")
            local_dir = self.model_cache_dir/hf_model_name
            print(local_dir)
            if local_dir.exists():
                logger.info(f"使用本地缓存模型")
                self.model = SentenceTransformer(
                    str(local_dir),
                    device=device
                )
            else:
                logger.info(f"从HuggingFace下载模型...")
                self.model = SentenceTransformer(
                    hf_model_name,
                    device=device
                )
                self.model.save(str(local_dir))
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ 模型加载成功 (维度: {self.dimension}, 设备: {self.model.device})")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.info(f"\n💡 首次使用需要下载模型，请确保网络连接")
            logger.info(f"   如果访问慢，设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 1, 
               show_progress: bool = False, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress=False)[0]
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
        if len(vec1.shape) == 1 and len(vec2.shape) == 1:
            return float(np.dot(vec1, vec2))
        elif len(vec1.shape) == 2 and len(vec2.shape) == 1:
            return vec1 @ vec2
        elif len(vec1.shape) == 2 and len(vec2.shape) == 2:
            return vec1 @ vec2.T
        else:
            raise ValueError(f"不支持的向量形状: {vec1.shape}, {vec2.shape}")
    
    @classmethod
    def list_models(cls):
        print("\n" + "="*70)
        print("🎯 推荐的中文Embedding模型")
        print("="*70)
        for key, info in cls.RECOMMENDED_MODELS.items():
            print(f"\n📌 {key}")
            print(f"   HuggingFace: {info['hf_name']}")
            print(f"   大小: {info['size']} | 维度: {info['dim']}")
            print(f"   说明: {info['description']}")
        print("\n" + "="*70)


# ==================== Embedding检索器（带缓存） ====================
class EmbeddingRetriever:
    """基于Embedding的工具检索器（支持缓存）"""
    
    def __init__(self, tools_registry: Dict, embedding_model: LocalCachedEmbedding, 
                 cache_dir: str = "./cache"):
        self.tools = tools_registry
        self.embedding = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.tool_embeddings = {}
        self.tool_names = []
        
        # 尝试加载缓存，如果失败则重新构建
        if not self._load_cache():
            self._build_index()
            self._save_cache()
    
    
    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        model_name = self.embedding.model_info.get("hf_name", "unknown").split('/')[-1].replace("/", "_")
        dimension = self.embedding.dimension
        return self.cache_dir / f"tool_embeddings_{model_name}_{dimension}.pkl"
    
    def _save_cache(self):
        """保存embedding到缓存"""
        cache_path = self._get_cache_path()
        
        try:
            cache_data = {
                "tool_embeddings": self.tool_embeddings,
                "tool_names": self.tool_names,
                "model_info": self.embedding.model_info,
                "dimension": self.embedding.dimension,
                "created_at": datetime.now().isoformat()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"💾 已保存embedding缓存: {cache_path}")
            logger.info(f"   缓存大小: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.warning(f"⚠️ 保存缓存失败: {e}")
    
    def _load_cache(self) -> bool:
        """从缓存加载embedding"""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            logger.info(f"📂 缓存文件不存在，将重新构建索引")
            return False
        
        try:
            logger.info(f"🔍 发现缓存文件: {cache_path}")
            logger.info(f"   缓存大小: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存
            cached_model = cache_data.get("model_info", {}).get("hf_name")
            current_model = self.embedding.model_info.get("hf_name")
            
            if cached_model != current_model:
                logger.warning(f"⚠️ 模型不匹配（缓存: {cached_model}, 当前: {current_model}），重新构建")
                return False
            
            # 加载缓存数据
            self.tool_embeddings = cache_data["tool_embeddings"]
            self.tool_names = cache_data["tool_names"]
            
            logger.info(f"✅ 成功加载缓存")
            logger.info(f"   工具数量: {len(self.tool_names)}")
            logger.info(f"   向量维度: {cache_data.get('dimension', 'unknown')}")
            logger.info(f"   创建时间: {cache_data.get('created_at', 'unknown')}")
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ 加载缓存失败: {e}，重新构建索引")
            return False
    
    def _build_index(self):
        """构建工具向量索引"""
        logger.info("🔨 构建工具向量索引...")
        
        tool_texts = []
        self.tool_names = []
        
        for name, info in self.tools.items():
            schema = info.get("schema", {}).get("function", {})
            desc = schema.get("description", "")
            params = schema.get("parameters", {}).get("properties", {})
            
            # 组合描述
            full_desc = f"{name}: {desc}"
            if params:
                param_desc = ", ".join([f"{k}({v.get('description', '')})" 
                                      for k, v in params.items()])
                full_desc += f" 参数: {param_desc}"
            
            tool_texts.append(full_desc)
            self.tool_names.append(name)
        
        # 批量编码
        logger.info(f"   正在编码 {len(tool_texts)} 个工具...")
        embeddings = self.embedding.encode(tool_texts, show_progress=True)
        
        for name, emb in zip(self.tool_names, embeddings):
            self.tool_embeddings[name] = emb
        
        logger.info(f"✅ 已索引 {len(self.tool_embeddings)} 个工具")
    
    
    def clear_cache(self):
        """清除缓存"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"🗑️ 已清除缓存: {cache_path}")
        else:
            logger.info(f"📂 无缓存文件")
    
    def rebuild_index(self):
        """强制重建索引"""
        logger.info("🔄 强制重建索引...")
        self._build_index()
        self._save_cache()
    
    def retrieve_top_k(self, question: str, top_k: int = 50) -> List[str]:
        """检索Top-K最相关的工具"""
        logger.info(f"\n🔍 Embedding检索阶段")
        logger.info(f"   问题: {question}")
        logger.info(f"   目标: Top-{top_k} 工具")
        
        # 编码问题
        question_embedding = self.embedding.encode_single(question)
        
        # 计算相似度
        similarities = {}
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = LocalCachedEmbedding.cosine_similarity(
                question_embedding, tool_embedding
            )
            similarities[tool_name] = float(similarity)
        
        # 排序返回Top-K
        sorted_tools = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_tools = [name for name, _ in sorted_tools[:top_k]]
        
        logger.info(f"\n✅ 检索到 Top-{len(top_k_tools)} 工具:")
        for i, (name, sim) in enumerate(sorted_tools[:10], 1):
            logger.info(f"   {i}. {name} (相似度: {sim:.4f})")
        if len(sorted_tools) > 10:
            logger.info(f"   ... 还有 {len(sorted_tools) - 10} 个工具")
        
        return top_k_tools


