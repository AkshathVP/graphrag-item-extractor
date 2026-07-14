from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    label: str = Field(..., description="Category: PERSON, TEAM, PROJECT, or CONCEPT")

class Relationship(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(..., description="Relationship type")

class ActionItem(BaseModel):
    task_id: str = Field(..., description="Unique task identifier, e.g., TASK_001")
    description: str = Field(..., description="Actionable description of the task")
    assignee: str = Field(..., description="Person or cross-functional team responsible")
    deadline: Optional[str] = Field(None, description="Due date or sprint timeframe")
    status: str = Field("Pending", description="Current execution status")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs or entities this task depends on")

class MeetingExtraction(BaseModel):
    meeting_id: str
    action_items: List[ActionItem]
    entities: List[Entity]
    relationships: List[Relationship]
