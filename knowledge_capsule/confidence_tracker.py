"""
Confidence Tracker - 基于贝叶斯更新的置信度追踪器
"""

from typing import Dict, List, Optional
from collections import defaultdict
import math

class ConfidenceTracker:
    """实验假设置信度追踪"""
    
    def __init__(self, base_rate: float = 0.3):
        self.base_rate = base_rate  # 基线成功率
        self.hypotheses: Dict[str, dict] = {}
        self.history: List[dict] = []
    
    def add_hypothesis(self, hypo_id: str, description: str, 
                       prior_knowledge: List[dict] = None) -> dict:
        """添加新假设"""
        # 基于先验知识计算初始置信度
        prior_confidence = self.base_rate
        if prior_knowledge:
            for pk in prior_knowledge:
                prior_confidence *= pk.get('confidence', 0.5)
            prior_confidence = min(1.0, prior_confidence * 2)
        
        self.hypotheses[hypo_id] = {
            "id": hypo_id,
            "description": description,
            "confidence": prior_confidence,
            "prior_knowledge": prior_knowledge or [],
            "evidence": [],
            "created_at": None
        }
        return self.hypotheses[hypo_id]
    
    def update(self, hypo_id: str, evidence: dict) -> float:
        """根据新证据更新置信度 (贝叶斯更新)"""
        if hypo_id not in self.hypotheses:
            return 0.0
        
        hypo = self.hypotheses[hypo_id]
        
        # P(H|E) = P(E|H) * P(H) / P(E)
        # 简化：使用比例更新
        p_h = hypo["confidence"]
        
        if evidence.get("success"):
            # 成功证据：提高置信度
            likelihood = 0.8
        else:
            # 失败证据：降低置信度
            likelihood = 0.3
        
        # 贝叶斯更新
        p_h_new = (likelihood * p_h) / (likelihood * p_h + (1 - likelihood) * (1 - p_h))
        
        hypo["confidence"] = p_h_new
        hypo["evidence"].append(evidence)
        self.history.append({"hypo_id": hypo_id, **evidence})
        
        return p_h_new
    
    def get_top_hypotheses(self, n: int = 5) -> List[dict]:
        """获取top N最可信的假设"""
        sorted_hypos = sorted(
            self.hypotheses.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )
        return sorted_hypos[:n]
    
    def predict_success_probability(self, hypo_id: str) -> float:
        """预测假设的成功概率"""
        if hypo_id not in self.hypotheses:
            return self.base_rate
        return self.hypotheses[hypo_id]["confidence"]
