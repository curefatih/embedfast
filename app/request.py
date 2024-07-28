from typing import List
from pydantic import BaseModel


class DocumentRequest(BaseModel):
    documents: List[str]
