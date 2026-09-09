"""Structured output contract for the Reflect Agent."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MetricSnapshot(BaseModel):
    candidate_id: str
    backend: Optional[str] = None
    mutations: List[str] = Field(default_factory=list)
    iptm: Optional[float] = None
    plddt: Optional[float] = None
    ranking_score: Optional[float] = None
    loglikelihood: Optional[float] = None
    cdr_rmsd: Optional[float] = None


class ImprovementCounts(BaseModel):
    iptm: int = Field(ge=0)
    plddt: int = Field(ge=0)
    ranking_score: int = Field(ge=0)


class PerformanceBreakdown(BaseModel):
    category: Literal["backend", "cdr", "mechanism"]
    label: str
    candidate_count: int = Field(ge=0)
    improved_count: int = Field(ge=0)
    summary: str


class PerformanceSummary(BaseModel):
    parent: MetricSnapshot
    best_candidate: Optional[MetricSnapshot] = None
    worst_candidate: Optional[MetricSnapshot] = None
    improvement_counts: ImprovementCounts
    breakdowns: List[PerformanceBreakdown] = Field(default_factory=list)
    observation: str


class PatternGroup(BaseModel):
    cdr: str
    mechanism: str
    candidate_ids: List[str]
    mutations: List[str]
    metric_evidence: List[str]
    rationale: str
    reusable_lesson: str = ""
    future_constraints: List[str] = Field(default_factory=list)


class CDRContactQuality(BaseModel):
    cdr: str
    residue_range: str
    epitope_contact_count: int = Field(ge=0)
    mean_interface_plddt: Optional[float] = None
    mean_interface_pae: Optional[float] = None
    hydrogen_bonds: int = Field(default=0, ge=0)
    salt_bridges: int = Field(default=0, ge=0)
    hydrophobic_contacts: int = Field(default=0, ge=0)
    status: str


class InterfaceResidueDirective(BaseModel):
    residue_anchor: str
    action: Literal["keep", "optimize", "avoid"]
    target_contacts: List[str] = Field(default_factory=list)
    reason: str


class UncontactedHotspot(BaseModel):
    hotspot: str
    nearest_cdr_residue: Optional[str] = None
    distance_angstrom: Optional[float] = None


class HotspotCoverage(BaseModel):
    contacted: List[str] = Field(default_factory=list)
    not_contacted: List[UncontactedHotspot] = Field(default_factory=list)


class StructureEpitopeAnalysis(BaseModel):
    available: bool
    cdr_contacts: List[CDRContactQuality] = Field(default_factory=list)
    interface_residues: List[InterfaceResidueDirective] = Field(default_factory=list)
    hotspot_coverage: HotspotCoverage = Field(default_factory=HotspotCoverage)
    summary: str


class EvidenceSection(BaseModel):
    available: bool
    summary: str
    evidence: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class NextCycleRecommendation(BaseModel):
    priority: int = Field(ge=1)
    cdr: str
    action: str
    evidence_basis: str
    confidence: Literal["high", "medium", "low"]
    mutation_budget: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)


class ReflectOutput(BaseModel):
    """Structured reflection consumed by the next design cycle."""

    performance_summary: PerformanceSummary
    successful_patterns: List[PatternGroup] = Field(default_factory=list)
    failure_patterns: List[PatternGroup] = Field(default_factory=list)
    structure_and_epitope: StructureEpitopeAnalysis
    sequence_plausibility: EvidenceSection
    developability: EvidenceSection
    search_trajectory_analysis: EvidenceSection
    evolutionary_tree_analysis: EvidenceSection
    next_cycle_recommendations: List[NextCycleRecommendation] = Field(default_factory=list)

    def to_design_directives(self) -> str:
        """Return only compact, actionable evidence for downstream agents."""
        structure = self.structure_and_epitope
        lines = ["### DESIGN DIRECTIVES", "CDR_CONTACTS:"]
        if structure.available and structure.cdr_contacts:
            for item in structure.cdr_contacts[:6]:
                metrics = [f"contacts={item.epitope_contact_count}"]
                if item.mean_interface_plddt is not None:
                    metrics.append(f"interface_pLDDT={item.mean_interface_plddt:g}")
                if item.mean_interface_pae is not None:
                    metrics.append(f"interface_pAE={item.mean_interface_pae:g}")
                metrics.extend(
                    [
                        f"H-bond={item.hydrogen_bonds}",
                        f"salt-bridge={item.salt_bridges}",
                        f"hydrophobic={item.hydrophobic_contacts}",
                    ]
                )
                lines.append(f"- {item.cdr} {item.residue_range} | " + " | ".join(metrics))
        else:
            lines.append("- unavailable")

        lines.append("INTERFACE_RESIDUES:")
        if structure.available and structure.interface_residues:
            for item in structure.interface_residues[:8]:
                contacts = f" | contacts={','.join(item.target_contacts)}" if item.target_contacts else ""
                lines.append(f"- {item.action.upper()} {item.residue_anchor}{contacts} | {item.reason}")
        else:
            lines.append("- unavailable")

        lines.append("HOTSPOT_COVERAGE:")
        contacted = structure.hotspot_coverage.contacted if structure.available else []
        lines.append(f"- CONTACTED {', '.join(contacted) if contacted else 'none observed'}")
        for item in structure.hotspot_coverage.not_contacted if structure.available else []:
            nearest = f" | nearest_CDR={item.nearest_cdr_residue}" if item.nearest_cdr_residue else ""
            distance = f" | distance={item.distance_angstrom:g}A" if item.distance_angstrom is not None else ""
            lines.append(f"- NOT_CONTACTED {item.hotspot}{nearest}{distance}")

        lines.append("PATTERN_EVIDENCE:")
        for item in self.successful_patterns[:3]:
            lines.append(f"- SUCCESS {item.cdr}/{item.mechanism} | {item.reusable_lesson or item.rationale}")
        for item in self.failure_patterns[:3]:
            constraint = "; ".join(item.future_constraints) or item.reusable_lesson or item.rationale
            lines.append(f"- AVOID {item.cdr}/{item.mechanism} | {constraint}")

        lines.append("SEARCH_HISTORY:")
        lines.append(f"- {self.search_trajectory_analysis.summary}")
        for item in self.search_trajectory_analysis.constraints[:4]:
            lines.append(f"- CONSTRAINT {item}")

        lines.append("EVOLUTIONARY_HISTORY:")
        lines.append(f"- {self.evolutionary_tree_analysis.summary}")
        for item in self.evolutionary_tree_analysis.constraints[:4]:
            lines.append(f"- CONSTRAINT {item}")

        lines.append("NEXT_PRIORITY:")
        for item in sorted(self.next_cycle_recommendations, key=lambda value: value.priority)[:6]:
            budget = f" | budget={item.mutation_budget}" if item.mutation_budget else ""
            constraints = f" | constraints={'; '.join(item.constraints)}" if item.constraints else ""
            lines.append(
                f"- P{item.priority} {item.cdr}: {item.action} | "
                f"evidence={item.evidence_basis} | confidence={item.confidence}{budget}{constraints}"
            )
        return "\n".join(lines)
