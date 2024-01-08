from __future__ import annotations

from typing import Any, Self, Callable, NewType
from types import MethodType
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class TokenBase:
    start: int
    end: int
    field_content: str


@dataclass
class ActiveToken(TokenBase):
    instruction: str


@dataclass
class InactiveToken(TokenBase):
    pass


class LexerBase(ABC):

    class LexerError(Exception):
        pass

    @abstractmethod
    def tokenize(self, template: str) -> list[TokenBase]:
        pass


class SimpleLexer(LexerBase):

    def tokenize(self, template: str) -> list[TokenBase]:
        LEFT_DELIMITER = '{{'
        RIGHT_DELIMITER = '}}'
        tokens = []
        prev_end = 0
        tag_reg = re.compile(LEFT_DELIMITER + r'(.+?)' + RIGHT_DELIMITER)

        for mach in tag_reg.finditer(template):
            if prev_end != mach.start():
                tokens.append(InactiveToken(
                    field_content=template[prev_end:mach.start()],
                    start=prev_end,
                    end=mach.start()))
            tokens.append(ActiveToken(
                instruction=mach.group(1),
                field_content=template[mach.start():mach.end()],
                start=mach.start(),
                end=mach.end()))
            prev_end = mach.end()

        if prev_end != len(template):
            tokens.append(InactiveToken(field_content=template[prev_end:], start=prev_end, end=len(template)))
        return tokens


# PARSER ========================================================


class Parser:
    ...


class TreeNode(ABC):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        self.parser = parser
        self.tokens = tokens
        self.fields_t = fields_t
        self.extra_context = extra_context

    @abstractmethod
    def make_tree(self, start: int) -> Self:
        pass

    @abstractmethod
    def render(self, context: Any) -> str:
        pass


class _FieldSignature:

    def __init__(self, owner: type[Field], signature: str, func: Field.FieldMethod):
        self.owner = owner
        self.signature = signature
        self.func = func

    def __get__(self, instance: Field, owner: type[Field]):
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(self, instruction: str, control: Field.ControlFlowInfo):
        return self.func(instruction, control)

    def match(self, text: str) -> bool:
        regex = re.compile(self.signature)
        return regex.match(text) is not None


class InitialField(_FieldSignature):
    pass


class MiddleField(_FieldSignature):
    pass


class FinalField(_FieldSignature):
    pass


class Field(ABC, TreeNode):

    @dataclass
    class ControlFlowInfo:
        context: Any
        extra_context: Any
        index: int

    FieldMethod = NewType('FieldMethod', Callable[[str, ControlFlowInfo], ControlFlowInfo])

    @classmethod
    def initial(cls, regex: str) -> Callable[[FieldMethod], InitialField]:

        def decorator(method: Field.FieldMethod) -> InitialField:
            tmp = InitialField(cls, regex, method)
            cls._initial_fields.append(tmp)
            return tmp
        return decorator

    @classmethod
    def middle(cls, regex: str) -> Callable[[FieldMethod], MiddleField]:
        def decorator(method: Field.FieldMethod) -> MiddleField:
            tmp = MiddleField(cls, regex, method)
            cls._middle_fields.append(tmp)
            return tmp
        return decorator

    @classmethod
    def final(cls, regex: str) -> Callable[[FieldMethod], FinalField]:
        def decorator(method: Field.FieldMethod) -> FinalField:
            tmp = FinalField(cls, regex, method)
            cls._final_fields.append(tmp)
            return tmp
        return decorator

    _initial_fields: list[InitialField] = []
    _middle_fields: list[MiddleField] = []
    _final_fields: list[FinalField] = []

    def __init__(self, tokens: list[TokenBase]):
        self.tokens = tokens
        self.inner_field: ContentNode = None

    @classmethod
    def initial_fields(cls) -> list[InitialField]:
        return cls._initial_fields.copy()

    @classmethod
    def middle_fields(cls) -> list[MiddleField]:
        return cls._initial_fields.copy()

    @classmethod
    def final_fields(cls) -> list[FinalField]:
        return cls._initial_fields.copy()

    @classmethod
    def all_fields(cls) -> list[_FieldSignature]:
        return cls.initial_fields() + cls.middle_fields() + cls.final_fields()

    @abstractmethod
    def start_context(self):
        pass

    @abstractmethod
    def check_context(self, signature: _FieldSignature):
        pass

    def make_tree(self, start: int) -> Self:
        self.start_context()
        current_token = self.tokens[start]
        assert isinstance(current_token, ActiveToken)

        match = next((f for f in self.all_fields() if f.match(current_token.instruction)), None)
        if match is None:
            raise Exception  # TODO: raise exception

        self.check_context(match)
        self.inner_field = ContentNode(self.parser, self.tokens, self.fields_t, self.extra_context)


class FieldNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.children: list[MiddleField] = None
        self.field = None

    def make_tree(self, start: int) -> Self:
        ...

    def render(self, context: Any) -> str:
        ...


class ContentNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.children: list[FieldNode | LeafNode] = None

    def make_tree(self, start: int) -> Self:
        ...

    def render(self, context: Any) -> str:
        ...


class LeafNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: str = None

    def make_tree(self, start: int) -> Self:
        ...

    def render(self, context: Any) -> str:
        ...


class EndNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.end_field: FinalField = None

    def make_tree(self, start: int) -> Self:
        ...

    def render(self, context: Any) -> str:
        ...
