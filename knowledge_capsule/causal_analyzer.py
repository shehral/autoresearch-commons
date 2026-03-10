"""
Causal Analyzer - 实验因果关系分析器
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json

class CausalAnalyzer:
    """分析超参数与实验结果的因果关系"""
    
    def __init__(self):
        # 因果图: variable -> {dependent_variables}
        self.causal_graph: Dict[str, Set[str]] = defaultdict(set)
        # 实验日志
        self.experiments: List[dict] = []
        # 变量关系
        self.relations: List[dict] = []
    
    def add_experiment(self, config: dict, result: float) -> None:
        """添加实验记录"""
        exp = {
            "config": config,
            "result": result,
            "timestamp": None
        }
        self.experiments.append(exp)
        self._analyze_relations(config, result)
    
    def _analyze_relations(self, config: dict, result: float) -> None:
        """分析配置与结果的关系"""
        for var, val in config.items():
            relation = {
                "variable": var,
                "value": val,
                "result": result,
                "type": self._classify_correlation(var, val, result)
            }
            self.relations.append(relation)
    
    def _classify_correlation(self, var: str, val, result: float) -> str:
        """分类变量与结果的相关性"""
        # 简化实现
        return "positive" if result < 1.0 else "neutral"
    
    def find_causal_variables(self) -> List[Tuple[str, float]]:
        """找出对结果影响最大的变量"""
        var_impact = defaultdict(list)
        
        for rel in self.relations:
            var = rel["variable"]
            var_impact[var].append(rel)
        
        # 计算每个变量的影响力
        impact_scores = []
        for var, rels in var_impact.items():
            # 简化：基于结果差异计算影响力
            results = [r["result"] for r in rels]
            if len(results) > 1:
                variance = max(results) - min(results)
                impact_scores.append((var, variance))
        
        return sorted(impact_scores, key=lambda x: x[1], reverse=True)
    
    def generate_report(self) -> dict:
        """生成因果分析报告"""
        top_vars = self.find_causal_variables()
        
        return {
            "total_experiments": len(self.experiments),
            "key_variables": [v[0] for v in top_vars[:5]],
            "variables_impact": dict(top_vars[:10]),
            "recommendations": self._generate_recommendations(top_vars)
        }
    
    def _generate_recommendations(self, top_vars: List[Tuple[str, float]]) -> List[str]:
        """基于分析生成建议"""
        recs = []
        for var, impact in top_vars[:3]:
            recs.append(f"重点关注 {var} 参数，变化幅度: {impact:.4f}")
        return recs
