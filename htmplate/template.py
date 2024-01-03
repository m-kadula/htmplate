from __future__ import annotations

from typing import Self, Any
from abc import ABC, abstractmethod
from pathlib import Path
import re


class FieldAbstract(ABC):

    class FieldCreationError(Exception):
        pass

    class FieldExecutionError(Exception):
        pass

    def __init__(self, field_content: str):
        self.field_content = field_content
        if not self.is_correct_field_content(field_content):
            raise self.FieldCreationError(f'Incorrect field content: {field_content}')

    @classmethod
    @abstractmethod
    def get_regex(cls) -> str:
        ...

    @classmethod
    def is_correct_field_content(cls, field_content: str) -> bool:
        return re.fullmatch(cls.get_regex(), field_content) is not None

    @classmethod
    def init(cls, field_content: str) -> Self:
        return cls(field_content)

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> str:
        ...


class DataField(FieldAbstract):

    regex = r'{%\s*([a-zA-Z0-9_]+)\s*%}'

    def __init__(self, field_content: str):
        super().__init__(field_content)
        match = re.fullmatch(self.regex, field_content)
        self.field_name = match.group(1)

    @classmethod
    def get_regex(cls) -> str:
        return cls.regex

    def execute(self, context: dict[str, Any]) -> tuple[str, bool]:
        value = context.get(self.field_name)
        if value is None:
            return self.field_content, False
        if not isinstance(value, str):
            raise self.FieldExecutionError(f'Value for field {self.field_name} is not a string')
        return value, True


class FileField(FieldAbstract):

    regex = r'{%\s*!file\s+([a-zA-Z0-9_.,~\\/-]+)\s*%}'

    def __init__(self, field_content: str):
        super().__init__(field_content)
        match = re.fullmatch(self.regex, field_content)
        self.file_path = Path(match.group(1)).resolve()
        if not self.file_path.exists():
            raise self.FieldCreationError(f'File {self.file_path} does not exist')

    @classmethod
    def get_regex(cls) -> str:
        return cls.regex

    def execute(self, context: dict[str, Any]) -> tuple[str, bool]:
        try:
            with open(self.file_path) as f:
                return f.read(), True
        except UnicodeDecodeError:
            raise self.FieldExecutionError(f'File {self.file_path} is not a text file')


class HTMLTemplate:
    def __init__(self, template: str):
        self.template = template

    def fields(self) -> frozenset[str]:
        ...

    def make(self, values: dict) -> Self:
        ...
