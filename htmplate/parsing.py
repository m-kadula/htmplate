from __future__ import annotations

from enum import Enum
from typing import Any, Callable, NewType
from types import MethodType
from dataclasses import dataclass, replace
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

    class ParsingError(Exception):
        pass

    def __init__(self, lexer: LexerBase, field_types: list[type[ControlField]]):
        self.lexer = lexer
        self.field_types = field_types

    def make_tree(self, template: str, extra_context: Any = None) -> ContentNode:
        tokens = self.lexer.tokenize(template)
        root = ContentNode(self, tokens, self.field_types, extra_context)
        out = root.make_tree(0)
        if out != len(tokens):
            raise Parser.ParsingError('Unexpected end of template')
        return root

    def parse(self, template: str, context: Any, **extra_context) -> str:
        tree = self.make_tree(template, extra_context)
        return tree.render(context)


class TreeNode(ABC):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        self.parser = parser
        self.tokens = tokens
        self.fields_t = fields_t
        self.extra_context = extra_context

    @abstractmethod
    def make_tree(self, start: int) -> int:
        pass

    @abstractmethod
    def render(self, context: Any) -> str:
        pass


class FieldSignature(ABC):

    @abstractmethod
    def match(self, text: str) -> bool:
        pass

    @abstractmethod
    def get_owner(self) -> type[Field]:
        pass


class ControlFieldSignature(FieldSignature):

    class FieldType(Enum):
        INITIAL = 0
        MIDDLE = 1
        FINAL = 2

    def __init__(self, f_type: FieldType, owner: type[ControlField], signature: str, func: ControlField.FieldMethod):
        self.owner = owner
        self.signature = signature
        self.func = func
        self.f_type = f_type

    def __get__(self, instance: ControlField, owner: type[ControlField]):
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(self, instruction: str, control: ControlField.ControlFlowInfo):
        return self.func(instruction, control)

    def match(self, text: str) -> bool:
        regex = re.compile(self.signature)
        return regex.match(text) is not None

    def get_owner(self) -> type[Field]:
        return self.owner


class SingleFieldSignature(FieldSignature):

    def __init__(self, owner: type[SingleField], signature: str):
        self.owner = owner
        self.signature = signature

    def match(self, text: str) -> bool:
        regex = re.compile(self.signature)
        return regex.match(text) is not None

    def get_owner(self) -> type[Field]:
        return self.owner


class Field(ABC, TreeNode):

    @classmethod
    @abstractmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        pass


class ControlField(ABC, Field):

    @dataclass
    class ControlFlowInfo:
        context: Any
        extra_context: Any
        index: int
        exit_next: bool = False

    FieldMethod = NewType('FieldMethod', Callable[[str, ControlFlowInfo], tuple[ControlFlowInfo, str | None]])

    @classmethod
    def initial(cls, regex: str) -> Callable[[FieldMethod], ControlFieldSignature]:

        def decorator(method: ControlField.FieldMethod) -> ControlFieldSignature:
            tmp = ControlFieldSignature(cls, ControlFieldSignature.FieldType.INITIAL, regex, method)
            cls._initial_fields.append(tmp)
            return tmp
        return decorator

    @classmethod
    def middle(cls, regex: str) -> Callable[[FieldMethod], ControlFieldSignature]:
        def decorator(method: ControlField.FieldMethod) -> ControlFieldSignature:
            tmp = ControlFieldSignature(cls, ControlFieldSignature.FieldType.MIDDLE, regex, method)
            cls._middle_fields.append(tmp)
            return tmp
        return decorator

    @classmethod
    def final(cls, regex: str) -> Callable[[FieldMethod], ControlFieldSignature]:
        def decorator(method: ControlField.FieldMethod) -> ControlFieldSignature:
            tmp = ControlFieldSignature(cls, ControlFieldSignature.FieldType.FINAL, regex, method)
            cls._final_fields.append(tmp)
            return tmp
        return decorator

    _initial_fields: list[ControlFieldSignature] = []
    _middle_fields: list[ControlFieldSignature] = []
    _final_fields: list[ControlFieldSignature] = []

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: list[tuple[ControlFieldSignature, ActiveToken, ContentNode]] = None

    @classmethod
    def initial_fields(cls) -> list[ControlFieldSignature]:
        return cls._initial_fields.copy()

    @classmethod
    def middle_fields(cls) -> list[ControlFieldSignature]:
        return cls._initial_fields.copy()

    @classmethod
    def final_fields(cls) -> list[ControlFieldSignature]:
        return cls._initial_fields.copy()

    @classmethod
    def all_fields(cls) -> list[ControlFieldSignature]:
        return cls.initial_fields() + cls.middle_fields() + cls.final_fields()

    def get_matching_signature(self, instruction: str) -> ControlFieldSignature | None:
        for field in self.all_fields():
            if field.match(instruction):
                return field
        raise None

    @abstractmethod
    def start_context(self):
        pass

    @abstractmethod
    def check_context(self, signature: ControlFieldSignature):
        pass

    def make_tree(self, start: int) -> int:
        self.start_context()
        index = start
        out: list[tuple[ControlFieldSignature, ActiveToken, ContentNode]] = []

        while index < len(self.tokens):
            current_token = self.tokens[index]
            assert isinstance(current_token, ActiveToken)

            match = self.get_matching_signature(current_token.instruction)
            if match is None:
                raise Parser.ParsingError(f'No field found for {current_token.instruction}')

            self.check_context(match)
            if match.f_type in [ControlFieldSignature.FieldType.INITIAL, ControlFieldSignature.FieldType.MIDDLE]:
                content = ContentNode(self.parser, self.tokens, self.fields_t, self.extra_context)
                index = content.make_tree(index + 1)
                out.append((match, current_token, content))
            else:
                out.append((match, current_token, None))
                self.content = out
                return index + 1

        raise Parser.ParsingError('Unexpected end of template')

    def render(self, context: Any) -> str:
        out: list[str] = []
        control = self.ControlFlowInfo(context, self.extra_context, 0)
        while not control.exit_next:
            signature, token, node = self.content[control.index]
            control, inner_render = signature(token.instruction, replace(control))
            if inner_render is not None:
                out.append(inner_render)
        return ''.join(out)


class SingleField(ABC, Field):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: ActiveToken = None

    @classmethod
    @abstractmethod
    def get_regex(cls) -> str:
        ...

    @classmethod
    def get_field(cls) -> ControlFieldSignature:
        return SingleFieldSignature(cls, cls.get_regex())

    @classmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        if cls.get_field().match(instruction):
            return cls.get_field()
        return None

    @abstractmethod
    def execute_field(self, context: Any) -> str:
        ...

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, ActiveToken)
        if not self.get_field().match(token.instruction):
            raise Parser.ParsingError(f'No field found for {token.instruction}')
        self.content = token
        return start + 1

    def render(self, context: Any) -> str:
        return self.execute_field(context)


class ContentNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.children: list[Field | LeafNode | ContentNode] = None

    def _get_field(self, instruction: str) -> ControlFieldSignature:
        for field_t in self.fields_t:
            tmp = field_t.get_matching_signature(instruction)
            if tmp is not None:
                return tmp
        raise Parser.ParsingError(f'No field found for {instruction}')

    def make_tree(self, start: int) -> int:
        out: list[Field | LeafNode | ContentNode] = []

        current = start
        while current < len(self.tokens):
            token = self.tokens[current]

            if isinstance(token, ActiveToken):
                sig: FieldSignature = self._get_field(token.instruction)
                if isinstance(sig, ControlFieldSignature):
                    if sig.f_type in [ControlFieldSignature.FieldType.MIDDLE, ControlFieldSignature.FieldType.FINAL]:
                        self.children = out
                        return current
                field = sig.get_owner()(self.parser, self.tokens, self.fields_t, self.extra_context)
                current = field.make_tree(current)
                out.append(field)

            elif isinstance(token, InactiveToken):
                child = LeafNode(self.parser, self.tokens, self.fields_t, self.extra_context)
                current = child.make_tree(current)
                out.append(child)

            else:
                raise TypeError(f'Unknown token type {type(token)}')

        self.children = out
        return current

    def render(self, context: Any) -> str:
        out: list[str] = []
        for child in self.children:
            out.append(child.render(context))
        return ''.join(out)


class LeafNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: InactiveToken = None

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, InactiveToken)
        self.content = token
        return start + 1

    def render(self, context: Any) -> str:
        return self.content.field_content
