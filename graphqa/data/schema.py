"""Triple/GraphSample 데이터 클래스 + 파싱 유틸."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# (ENTk) placeholder 또는 우리가 생성하는 ?Xk 형식 모두 unknown 으로 간주
UNKNOWN_PATTERN = re.compile(r"^\(?ENT\d+\)?$|^\?[A-Za-z0-9_]+$", re.IGNORECASE)


def is_unknown(token: str) -> bool:
    """문자열이 UNKNOWN/placeholder 슬롯이면 True."""
    if not token:
        return True
    return bool(UNKNOWN_PATTERN.match(token.strip()))


def parse_triple(triple_sent: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """`H [SEP] R [SEP] T [PREP] context` 문자열 → (h, r, t, context)."""
    if not triple_sent or "[SEP]" not in triple_sent:
        return None
    parts = triple_sent.split(" [SEP] ")
    if len(parts) < 3:
        return None
    head = parts[0].strip()
    rel = parts[1].strip()
    tail_block = " [SEP] ".join(parts[2:]).strip()
    context: Optional[str] = None
    if " [PREP] " in tail_block:
        tail, context = tail_block.split(" [PREP] ", 1)
        tail = tail.strip()
        context = context.strip()
    else:
        tail = tail_block
    return head, rel, tail, context


@dataclass
class Triple:
    """정규화된 triple 표현."""
    head: str
    relation: str
    tail: str
    context: Optional[str] = None
    raw: Optional[str] = None

    @classmethod
    def from_str(cls, triple_sent: str) -> Optional["Triple"]:
        parsed = parse_triple(triple_sent)
        if not parsed:
            return None
        h, r, t, c = parsed
        return cls(head=h, relation=r, tail=t, context=c, raw=triple_sent)

    @property
    def head_unknown(self) -> bool:
        return is_unknown(self.head)

    @property
    def tail_unknown(self) -> bool:
        return is_unknown(self.tail)

    def to_text(self) -> str:
        """임베딩용 자연어 변환 (placeholder 제거)."""
        h = "" if self.head_unknown else self.head
        t = "" if self.tail_unknown else self.tail
        s = " ".join(x for x in (h, self.relation, t) if x)
        if self.context:
            s = f"{s} {self.context}"
        return s.strip()

    def __repr__(self) -> str:
        return f"Triple({self.head} [SEP] {self.relation} [SEP] {self.tail})"


@dataclass
class GraphStep:
    """multi-hop reasoning의 한 step."""
    step_index: int
    step_text: str
    triples: List[Triple] = field(default_factory=list)


@dataclass
class StepEvidence:
    """SearchR1 한 step에서 관측된 think/doc evidence triples."""
    step_index: int
    query: str = ""
    think_text: str = ""
    think_triples: List[Triple] = field(default_factory=list)
    doc_triples: List[Triple] = field(default_factory=list)
    doc_texts: List[str] = field(default_factory=list)


@dataclass
class GraphSample:
    """하나의 질문에 대해 4가지 그래프 + GT를 묶은 단위.

    - Q: question_graph (UNKNOWN 슬롯 포함)
    - T: reasoning(think) graph (LLM 추론 triples)
    - Sr: search graph (검색 키워드를 placeholder triple로 변환)
    - D: doc_graph (검색된 문서에서 추출된 fact triples)
    """

    uid: str
    question: str
    answer: str
    answer_aliases: List[str] = field(default_factory=list)
    num_hops: int = 0
    dataset: str = ""

    # 4가지 그래프
    Q_def: List[Triple] = field(default_factory=list)
    Q: List[Triple] = field(default_factory=list)
    T: List[Triple] = field(default_factory=list)
    T_steps: List[GraphStep] = field(default_factory=list)
    step_evidence: List[StepEvidence] = field(default_factory=list)
    Sr: List[Triple] = field(default_factory=list)
    D: List[Triple] = field(default_factory=list)

    # 부가정보
    search_queries: List[str] = field(default_factory=list)
    predicted_answer: Optional[str] = None
    gold_id_list: List[str] = field(default_factory=list)

    @property
    def has_all_graphs(self) -> bool:
        return bool(self.Q) and bool(self.T) and bool(self.Sr) and bool(self.D)

    def summary(self) -> str:
        return (
            f"GraphSample(dataset={self.dataset}, uid={self.uid[:8]}..., "
            f"|Q|={len(self.Q)}+{len(self.Q_def)}def, |T|={len(self.T)}, "
            f"|Sr|={len(self.Sr)}, |D|={len(self.D)}, hops={self.num_hops})"
        )
