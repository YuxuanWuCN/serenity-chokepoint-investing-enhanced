import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TaggedStatement:
    tag_type: str  # FACT, OPINION, INFERENCE, TRIANGULATED_FACT
    source_or_holder: str
    content: str
    is_valid: bool

class FOITagger:
    """
    Layer 1 FOI (Fact, Opinion, Inference) parser and downgrading system.
    Parses and enforces tagging standards:
    - [FACT:source]
    - [OPINION:holder]
    - [INFERENCE:chain]
    - [TRIANGULATED_FACT]
    """
    PATTERN = re.compile(r'\[(FACT|OPINION|INFERENCE|TRIANGULATED_FACT)(?::([^\]]+))?\]\s*([^\[]+)')

    def parse_text(self, text: str) -> List[TaggedStatement]:
        results = []
        matches = self.PATTERN.findall(text)
        for tag, meta, content in matches:
            content_clean = content.strip()
            is_valid = len(content_clean) > 0
            results.append(TaggedStatement(
                tag_type=tag,
                source_or_holder=meta.strip() if meta else "",
                content=content_clean,
                is_valid=is_valid
            ))
        return results

    def validate_document(self, text: str) -> Dict[str, Any]:
        tagged = self.parse_text(text)
        facts = [t for t in tagged if t.tag_type in ("FACT", "TRIANGULATED_FACT")]
        opinions = [t for t in tagged if t.tag_type == "OPINION"]
        inferences = [t for t in tagged if t.tag_type == "INFERENCE"]

        # Calculate coverage and metrics
        total_tagged = len(tagged)
        has_triangulation = any(t.tag_type == "TRIANGULATED_FACT" for t in tagged)
        avg_credibility = sum(self.calculate_credibility_score(t.source_or_holder, t.tag_type) for t in tagged) / total_tagged if total_tagged > 0 else 0.0

        return {
            "total_statements": total_tagged,
            "fact_count": len(facts),
            "opinion_count": len(opinions),
            "inference_count": len(inferences),
            "has_triangulation": has_triangulation,
            "is_grounded": len(facts) > 0 and len(facts) >= len(opinions),
            "credibility_score": round(avg_credibility, 2),
            "statements": [t.__dict__ for t in tagged]
        }

    @staticmethod
    def calculate_credibility_score(source_or_holder: str, tag_type: str) -> float:
        """
        Calculate evidence credibility score (0-100) based on source taxonomy:
        - Official Filings / Regulators / Primary Hard Data / Triangulated: 90 - 100
        - Financial Terminals (Wind, Bloomberg, Choice): 80 - 90
        - Brokerage Research / Sell-side Consensus: 60 - 75
        - Rumors / Social Media / Generic Opinion: 30 - 50
        """
        if tag_type == "TRIANGULATED_FACT":
            return 95.0
        
        src_lower = source_or_holder.lower()
        if any(k in src_lower for k in ["sec", "csrc", "annual_report", "stat_bureau", "customs", "audit"]):
            return 100.0
        elif any(k in src_lower for k in ["wind", "bloomberg", "reuters", "choice", "official"]):
            return 90.0
        elif any(k in src_lower for k in ["analyst", "broker", "research", "survey", "cicc", "citic"]):
            return 70.0
        elif tag_type == "FACT":
            return 80.0
        elif tag_type == "INFERENCE":
            return 60.0
        else: # OPINION
            return 40.0

    @staticmethod
    def extract_foi_prompt_template(raw_text: str) -> str:
        """
        Generate structured LLM extraction prompt for FinGPT / DeepSeek.
        """
        return f"""你是一名资深买方量化投研专家。请严格按照 [FOI] 证据分级体系从以下文本中提取关键声明并分类：
- [FACT:数据源]: 可验证的客观事实、财务数据或官方声明（如公司财报、海关进出口、权威终端数据）。
- [OPINION:持有者]: 分析师预测、管理层指引、市场观点或情绪。
- [INFERENCE:推导逻辑]: 基于事实推导出的供应链因果逻辑或投资假设。
- [TRIANGULATED_FACT]: 经由两个以上独立信息源互为印证的卡点事实。

待分析文本：
\"\"\"{raw_text}\"\"\"

请严格输出 JSON 格式（List[Dict]）：
[
  {{"tag_type": "FACT", "source_or_holder": "公司公告", "content": "2025Q1营收增长45%"}},
  {{"tag_type": "OPINION", "source_or_holder": "中金分析师", "content": "预计未来三年复合增速30%"}}
]"""

    def parse_llm_json_response(self, items: List[Dict[str, Any]]) -> List[TaggedStatement]:
        """
        Parse structured JSON response from LLM into TaggedStatements.
        """
        results = []
        for item in items:
            tag = item.get("tag_type", "FACT").upper()
            if tag not in ("FACT", "OPINION", "INFERENCE", "TRIANGULATED_FACT"):
                tag = "FACT"
            source = item.get("source_or_holder", "")
            content = item.get("content", "").strip()
            results.append(TaggedStatement(
                tag_type=tag,
                source_or_holder=source,
                content=content,
                is_valid=len(content) > 0
            ))
        return results
