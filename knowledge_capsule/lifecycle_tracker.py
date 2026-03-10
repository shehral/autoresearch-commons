"""
Knowledge Capsule Lifecycle Tracker
基于记忆树理念，为实验假设添加生命周期管理
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

class LifecycleStage(Enum):
    """实验假设生命周期阶段"""
    SPROUT = "🌱"  # 萌芽 (confidence: 0.7)
    GREEN_LEAF = "🌿"  # 绿叶 (confidence >= 0.8)
    YELLOW_LEAF = "🍂"  # 黄叶 (confidence: 0.5-0.8)
    RED_LEAF = "🍁"  # 枯叶 (confidence < 0.3)
    SOIL = "🪨"  # 土壤 (archived)

class LifecycleTracker:
    """实验生命周期追踪器"""
    
    def __init__(self, decay_rate: float = 0.004):
        self.decay_rate = decay_rate  # 每天衰减
        self.experiments: Dict[str, dict] = {}
    
    def add_experiment(self, exp_id: str, hypothesis: str) -> dict:
        """添加新实验假设"""
        self.experiments[exp_id] = {
            "id": exp_id,
            "hypothesis": hypothesis,
            "stage": LifecycleStage.SPROUT.value,
            "confidence": 0.7,
            "created_at": datetime.now().isoformat(),
            "last_validated": None,
            "validation_count": 0,
            "success_count": 0
        }
        return self.experiments[exp_id]
    
    def validate(self, exp_id: str, success: bool) -> dict:
        """验证实验结果，更新置信度"""
        if exp_id not in self.experiments:
            return None
        
        exp = self.experiments[exp_id]
        exp["validation_count"] += 1
        exp["last_validated"] = datetime.now().isoformat()
        
        # 贝叶斯更新
        if success:
            exp["success_count"] += 1
            exp["confidence"] = min(1.0, exp["confidence"] + 0.05)
        else:
            exp["confidence"] = max(0.0, exp["confidence"] - 0.1)
        
        # 更新阶段
        exp["stage"] = self._get_stage(exp["confidence"])
        
        return exp
    
    def _get_stage(self, confidence: float) -> str:
        """根据置信度获取阶段"""
        if confidence >= 0.8:
            return LifecycleStage.GREEN_LEAF.value
        elif confidence >= 0.5:
            return LifecycleStage.YELLOW_LEAF.value
        elif confidence >= 0.3:
            return LifecycleStage.RED_LEAF.value
        else:
            return LifecycleStage.SOIL.value
    
    def decay_all(self) -> None:
        """对未验证的假设进行衰减"""
        now = datetime.now()
        for exp in self.experiments.values():
            if exp["last_validated"]:
                last = datetime.fromisoformat(exp["last_validated"])
                days_passed = (now - last).days
                if days_passed > 0:
                    exp["confidence"] = max(0.0, exp["confidence"] - self.decay_rate * days_passed)
                    exp["stage"] = self._get_stage(exp["confidence"])
    
    def get_active_experiments(self) -> list:
        """获取活跃实验（绿叶阶段）"""
        return [e for e in self.experiments.values() 
                if e["stage"] == LifecycleStage.GREEN_LEAF.value]
    
    def export(self) -> dict:
        """导出所有实验数据"""
        return self.experiments
