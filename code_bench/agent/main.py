
import os
import sys


current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import json
import random
import logging
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from financial_agent import FinancialAgent
os.makedirs('./logs', exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s',
    handlers=[
            logging.FileHandler(f'./logs/agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

def save_data(data, output_path):
    with open(output_path,'a',encoding='utf-8') as f:
        f.write(json.dumps(asdict(data), ensure_ascii=False) + '\n')


# ==================== 初始化与主函数 ====================
def main(
    tool_path: str = "./tools",
    embedding_cache_dir: str = "./cache",
    top_k: int = 20,
    output_path: str = "data/result/result_ablation",
    data_path: str = "/data/question/select_data_real_sample50_seed42.jsonl",
    setting: Optional[str] = None,
    execution_model_name: str = "ep-20251113093937-p4rll",
    extract_model_name: str = "ep-20251101221159-hhmrg",
):
    logger.info("\n" + "="*100)
    logger.info("🚀 初始化金融Agent系统")
    logger.info("="*100)

    finacial_agent = FinancialAgent(tool_path=tool_path, embedding_cache_dir=embedding_cache_dir, execution_model_name=execution_model_name, extract_model_name=extract_model_name)
    logger.info("✅ 初始化完成\n")
    
    logger.info("初始化数据")

    data_lt = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_lt.append(json.loads(line))

    random.seed(42)

    logger.info(f"加载{len(data_lt)}条数据")
    
    os.makedirs(output_path, exist_ok=True)
    settings = {
        "full": {"inject_attributes": True},
        "wo_injection": {"inject_attributes": False},
    }

    run_settings = {setting: settings[setting]} if setting else settings
    for setting_name, setting_cfg in run_settings.items():
        output_file = os.path.join(output_path, f"result_{execution_model_name}_{setting_name}.jsonl")
        for idx, per_data in tqdm(enumerate(data_lt), total=len(data_lt), desc=f"Processing {setting_name}"):
            result = finacial_agent.run(
                idx,
                per_data,
                top_k=top_k,
                max_steps=5,
                inject_attributes=setting_cfg["inject_attributes"],
                setting=setting_name
            )
            save_data(result, output_file)



# ==================== 测试入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinancialAgent on a dataset.")
    parser.add_argument("--tool_path", default="./tools")
    parser.add_argument("--embedding_cache_dir", default="./cache")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_path", default="/data/result/result_ablation")
    parser.add_argument(
        "--data_path",
        default="data/question/select_data_real_remove_duplicates.jsonl",
    )
    parser.add_argument("--setting", choices=["full", "wo_injection"], default=None)
    parser.add_argument(
        "--execution_model_name", default="ep-20251113093937-p4rll"
    )
    parser.add_argument(
        "--extract_model_name", default="ep-20251101221159-hhmrg"
    )
    args = parser.parse_args()

    main(
        tool_path=args.tool_path,
        embedding_cache_dir=args.embedding_cache_dir,
        top_k=args.top_k,
        output_path=args.output_path,
        data_path=args.data_path,
        setting=args.setting,
        execution_model_name=args.execution_model_name,
        extract_model_name=args.extract_model_name,
    )
    
    


    
    
    
