"""Core data models for psy-bench."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScoreType(str, Enum):
    """Types of scores in the evaluation."""
    DCS = "dcs"  # Delusion Confirmation Score
    HES = "hes"  # Harm Enablement Score
    SIS = "sis"  # Safety Intervention Score


class TestCase(BaseModel):
    """A test case for AI safety evaluation."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    theme: str = Field(..., description="Theme/category of delusions")
    condition: str = Field(..., description="Explicit or Implicit")
    harm_type: str = Field(..., description="Type of potential harm")
    prompts: List[str] = Field(..., description="List of prompts (typically 12)")
    
    @property
    def phases(self) -> Dict[str, List[str]]:
        """Get prompts organized by phase."""
        return {
            "initial_engagement": self.prompts[0:3],
            "pattern_seeking": self.prompts[3:6],
            "delusion_solidification": self.prompts[6:9],
            "behavioral_enactment": self.prompts[9:12]
        }
    
    @property
    def phase_count(self) -> int:
        """Number of phases (typically 4)."""
        return len(self.prompts) // 3


class Turn(BaseModel):
    """A single conversation turn."""
    number: int = Field(..., description="Turn number (1-based)")
    user_message: str = Field(..., description="User's message")
    bot_response: str = Field(..., description="Bot's response")
    scores: Dict[ScoreType, Optional[int]] = Field(
        default_factory=dict,
        description="Scores for this turn"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ScoreResult(BaseModel):
    """Result of a scoring operation."""
    score: int = Field(..., description="Numeric score")
    score_type: ScoreType = Field(..., description="Type of score")
    explanation: Optional[str] = Field(None, description="Explanation from judge")
    error: Optional[str] = Field(None, description="Error message if scoring failed")
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid score."""
        return self.error is None and self.score >= 0


class ExperimentResult(BaseModel):
    """Complete result from an experiment."""
    test_case: TestCase = Field(..., description="Test case that was run")
    model: str = Field(..., description="Model that was tested")
    turns: List[Turn] = Field(..., description="All conversation turns")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Experiment metadata"
    )
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        # Extract scores by type
        dcs_scores = []
        hes_scores = []
        sis_scores = []
        
        for turn in self.turns:
            if ScoreType.DCS in turn.scores and turn.scores[ScoreType.DCS] is not None:
                dcs_scores.append(turn.scores[ScoreType.DCS])
            if ScoreType.HES in turn.scores and turn.scores[ScoreType.HES] is not None:
                hes_scores.append(turn.scores[ScoreType.HES])
            if ScoreType.SIS in turn.scores and turn.scores[ScoreType.SIS] is not None:
                sis_scores.append(turn.scores[ScoreType.SIS])
        
        return {
            "total_turns": len(self.turns),
            "avg_dcs": sum(dcs_scores) / len(dcs_scores) if dcs_scores else 0,
            "avg_hes": sum(hes_scores) / len(hes_scores) if hes_scores else 0,
            "total_sis": sum(sis_scores),
            "max_dcs": max(dcs_scores) if dcs_scores else 0,
            "max_hes": max(hes_scores) if hes_scores else 0,
        }


class RunConfig(BaseModel):
    """Configuration for running an experiment."""
    start_turn: int = Field(1, ge=1, le=12, description="Starting turn")
    end_turn: int = Field(12, ge=1, le=12, description="Ending turn")
    judge_model: Optional[str] = Field(None, description="Model for scoring")
    verbose: bool = Field(True, description="Print progress")
    delay_between_turns: float = Field(0.5, description="Delay in seconds")
