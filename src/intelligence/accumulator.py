from typing import Any, Callable, Iterator, Optional, Sequence, Union
from src.domain.enums import EvidenceKind
from src.domain.evidence import Evidence


class EvidenceAccumulator:
    def __init__(
        self,
        target_id: Optional[str] = None,
        track_id: Optional[str] = None,
        location: Optional[tuple[float, float]] = None,
    ) -> None:
        self.target_id = target_id
        self.track_id = track_id
        self.location = location
        self._evidence: list[Evidence] = []
        self._evidence_by_id: dict[str, Evidence] = {}
        self._seen_provenance_ids: set[str] = set()
        self._seen_source_ids: set[str] = set()

    def _extract_source_id(
        self, evidence: Evidence, explicit_source_id: Optional[str] = None
    ) -> Optional[str]:
        if explicit_source_id:
            return explicit_source_id
        for ref in evidence.references:
            if ref.startswith("source_id="):
                return ref.split("=", 1)[1]
            if ref.startswith("detection_id="):
                return ref.split("=", 1)[1]
            if ref.startswith("candidate_id="):
                return ref.split("=", 1)[1]
        return None

    def add(
        self, evidence: Evidence, source_id: Optional[str] = None
    ) -> bool:
        if evidence.evidence_id in self._evidence_by_id:
            return False

        prov_id = (
            evidence.provenance.provenance_id
            if evidence.provenance
            else None
        )
        if prov_id and prov_id in self._seen_provenance_ids:
            return False

        extracted_src_id = self._extract_source_id(evidence, source_id)
        if extracted_src_id and extracted_src_id in self._seen_source_ids:
            return False

        if prov_id:
            self._seen_provenance_ids.add(prov_id)
        if extracted_src_id:
            self._seen_source_ids.add(extracted_src_id)

        self._evidence_by_id[evidence.evidence_id] = evidence
        self._evidence.append(evidence)
        return True

    def add_evidence(
        self, evidence: Evidence, source_id: Optional[str] = None
    ) -> bool:
        return self.add(evidence, source_id=source_id)

    def extend(self, evidences: Sequence[Evidence]) -> int:
        count = 0
        for ev in evidences:
            if self.add(ev):
                count += 1
        return count

    def add_many(self, evidences: Sequence[Evidence]) -> int:
        return self.extend(evidences)

    def get_by_kind(
        self, kind: Union[EvidenceKind, str]
    ) -> tuple[Evidence, ...]:
        target_kind = kind if isinstance(kind, EvidenceKind) else EvidenceKind(kind)
        return tuple(ev for ev in self._evidence if ev.kind == target_kind)

    def query(
        self,
        kind: Optional[Union[EvidenceKind, str]] = None,
        source: Optional[str] = None,
        target_id: Optional[str] = None,
        track_id: Optional[str] = None,
        predicate: Optional[Callable[[Evidence], bool]] = None,
    ) -> tuple[Evidence, ...]:
        results = self._evidence
        if kind is not None:
            target_kind = kind if isinstance(kind, EvidenceKind) else EvidenceKind(kind)
            results = [ev for ev in results if ev.kind == target_kind]
        if source is not None:
            results = [ev for ev in results if ev.source == source]
        if target_id is not None:
            results = [
                ev
                for ev in results
                if any(
                    ref == f"target_id={target_id}"
                    or ref.startswith(f"target_id={target_id}")
                    for ref in ev.references
                )
            ]
        if track_id is not None:
            results = [
                ev
                for ev in results
                if any(
                    ref == f"track_id={track_id}"
                    or ref.startswith(f"track_id={track_id}")
                    for ref in ev.references
                )
            ]
        if predicate is not None:
            results = [ev for ev in results if predicate(ev)]
        return tuple(results)

    def all_evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "track_id": self.track_id,
            "location": self.location,
            "total_items": len(self._evidence),
            "evidence_ids": [ev.evidence_id for ev in self._evidence],
            "kinds": {
                kind.value: len(self.get_by_kind(kind))
                for kind in {ev.kind for ev in self._evidence}
            },
        }

    def summary(self) -> dict[str, Any]:
        return self.to_dict()

    def clear(self) -> None:
        self._evidence.clear()
        self._evidence_by_id.clear()
        self._seen_provenance_ids.clear()
        self._seen_source_ids.clear()

    def __len__(self) -> int:
        return len(self._evidence)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._evidence)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return item in self._evidence_by_id
        if hasattr(item, "evidence_id"):
            return item.evidence_id in self._evidence_by_id
        return False

    def __getitem__(self, evidence_id: str) -> Evidence:
        return self._evidence_by_id[evidence_id]
