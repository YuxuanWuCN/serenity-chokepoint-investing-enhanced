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

        return {
            "total_statements": total_tagged,
            "fact_count": len(facts),
            "opinion_count": len(opinions),
            "inference_count": len(inferences),
            "has_triangulation": has_triangulation,
            "is_grounded": len(facts) > 0 and len(facts) >= len(opinions),
            "statements": [t.__dict__ for t in tagged]
        }
