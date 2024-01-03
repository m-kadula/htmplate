from __future__ import annotations

from typing import Self, Any
from abc import ABC, abstractmethod
from pathlib import Path
import re


class FieldAbstract(ABC):

    _registry: set[FieldAbstract] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry.add(cls)

    @staticmethod
    def get_registry() -> frozenset[FieldAbstract]:
        return frozenset(FieldAbstract._registry)

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

    def get_field_content(self) -> str:
        return self.field_content

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

    regex = r'{%\s*!file\s+([a-zA-Z0-9_.,~\\/ -]+)\s*%}'

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
        self._fields_cache: None | list[tuple[int, int, FieldAbstract]] = None

    def fields(self) -> list[FieldAbstract]:
        if self._fields_cache is None:
            self._fields_cache = []
            for field_cls in FieldAbstract.get_registry():
                for match in re.finditer(field_cls.get_regex(), self.template):
                    self._fields_cache.append((match.start(), match.end(), field_cls.init(match.group(0))))
            self._fields_cache.sort(key=lambda x: x[0])

        return self._fields_cache.copy()

    @classmethod
    def _make_inner(cls, instance: Self, values: dict[str, Any]) -> tuple[Self, bool]:
        out_template = []
        last_end = 0
        filled_anything = False

        for start, end, field in instance.fields():
            if start < last_end:
                raise ValueError('Overlapping fields')  # TODO: change to custom exception

            out_template.append(instance.template[last_end:start])
            value, is_correct = field.execute(values)
            if is_correct:
                out_template.append(value)
                filled_anything = True
            else:
                out_template.append(field.get_field_content())
            last_end = end

        out_template.append(instance.template[last_end:])
        return instance.__class__(''.join(out_template)), filled_anything

    def make(self, values: dict[str, Any], iter_limit: int = -1) -> Self:
        instance = self
        filled_anything = True
        iterations = 0

        while filled_anything and (iterations < iter_limit or iter_limit == -1):
            instance, filled_anything = self._make_inner(instance, values)
            iterations += 1
        if filled_anything:
            raise ValueError('Iteration limit reached')

        return instance
