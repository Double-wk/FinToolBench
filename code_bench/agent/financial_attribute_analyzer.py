import json
from typing import Dict
import logging
logger = logging.getLogger(__name__)

# ==================== 金融属性分析器（保持不变）====================
class FinancialAttributeAnalyzer:
    """金融属性分析器"""
    
    def __init__(self, tools_registry: Dict):
        self.tools = tools_registry
        self.tool_attributes = self._load_financial_attributes()
    
    def _load_financial_attributes(self) -> Dict[str, Dict]:
        attributes = {}
        for name, info in self.tools.items():
            try:
                attributes[name] = info.get("financial_tags", {})
            except Exception as e:
                logger.warning(f"无法加载 {name} 的属性: {e}")
                attributes[name] = {}

        logger.info(f"📋 已加载 {len(attributes)} 个工具）")
        return attributes
    
    
    def get_attributes(self, tool_name: str) -> Dict:
        return self.tool_attributes.get(tool_name, {})
