import glob
import os
from pathlib import Path
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders import SRTLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders.base import BaseLoader


class LectureLoader(BaseLoader):
    def __init__(self, 
                 lecture_file: str,
                 add_lecture_info: bool = False
                ):
        self.add_lecture_info = add_lecture_info
        self.lecture_file = lecture_file

    @classmethod
    def from_folder(cls, folder_name: str, **kwargs: Any) -> 'LectureLoader':
        return cls(folder_name, kwargs)

    def load(self) -> List[Document]:
        documents = []
        
        for file_name in Path(self.lecture_file).rglob('*'):
            
            file_path = Path(file_name)
            if not file_path.is_file(): continue
                
            with open(file_name, "r") as f:

                metadata = {}
                
                # Load the transcript data
                if file_name.suffix == ".srt":
                    srt_loader = SRTLoader(file_name)
                        
                    if self.add_lecture_info:
                        metadata["lecture_name"] = file_path.parent.name
                        metadata["source"] = file_path.stem
                        metadata["type"] = "transcript"

                    for doc in srt_loader.load():
                        doc.metadata.update(metadata)
                        documents.append(doc)

                elif file_name.suffix == ".txt":
                    txt_loader = TextLoader(file_path)

                    if self.add_lecture_info:
                        metadata["lecture_name"] = file_path.parent.name
                        metadata["source"] = file_path.stem
                        metadata["type"] = "transcript"

                    for doc in txt_loader.load():
                        doc.metadata.update(metadata)
                        documents.append(doc)

        return documents
