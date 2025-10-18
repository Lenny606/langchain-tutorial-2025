from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    original_paragraph: str = Field(description="Original paragraph")
    edited_paragraph: str = Field(description="Edited paragraph")
    feedback: int = Field(description="constructive feedback on original as score ")