from __future__ import annotations

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

    def __init__(self, lexer: LexerBase, field_types: list[type[Field]]):
        self.lexer = lexer
        self.field_types = field_types

    def make_tree(self, template: str, extra_context: Any = None) -> ContentNode:
        tokens = self.lexer.tokenize(template)
        root = ContentNode(self, tokens, self.field_types, extra_context)
        root.make_tree(0)
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
        exit_next: bool = False

    FieldMethod = NewType('FieldMethod', Callable[[str, ControlFlowInfo], tuple[ControlFlowInfo, str | None]])

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

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: list[tuple[_FieldSignature, ActiveToken, ContentNode]] = None

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

    def make_tree(self, start: int) -> int:
        self.start_context()
        index = start
        out: list[tuple[_FieldSignature, ActiveToken, ContentNode]] = []

        while index < len(self.tokens):
            current_token = self.tokens[index]
            assert isinstance(current_token, ActiveToken)

            match = next((f for f in self.all_fields() if f.match(current_token.instruction)), None)
            if match is None:
                raise Parser.ParsingError(f'No field found for {current_token.instruction}')

            self.check_context(match)
            if isinstance(match, (InitialField, MiddleField)):
                content = ContentNode(self.parser, self.tokens, self.fields_t, self.extra_context)
                index = content.make_tree(index)
                out.append((match, current_token, content))
            else:
                out.append((match, current_token, None))
                self.content = out
                return index + 1

        self.check_context(None)
        self.content = out
        return index

    def render(self, context: Any) -> str:
        out: list[str] = []
        control = self.ControlFlowInfo(context, self.extra_context, 0)
        while not control.exit_next:
            signature, token, node = self.content[control.index]
            control, inner_render = signature(token.instruction, replace(control))
            if inner_render is not None:
                out.append(inner_render)
        return ''.join(out)


class ContentNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.children: list[Field | LeafNode | ContentNode] = None

    def _get_field(self, instruction: str) -> type[Field]:
        for field_t in self.fields_t:
            fields = field_t.all_fields()
            tmp = next((f for f in fields if f.match(instruction)), None)
            if tmp is not None:
                return field_t
        raise Exception  # TODO: raise exception

    def make_tree(self, start: int) -> int:
        out: list[Field | LeafNode | ContentNode] = []

        current = start
        while current < len(self.tokens):
            token = self.tokens[current]

            if isinstance(token, ActiveToken):
                field_t: type[Field] = self._get_field(token.instruction)
                if isinstance(field_t, (FinalField, MiddleField)):
                    self.children = out
                    return current
                field = field_t(self.parser, self.tokens, self.fields_t, self.extra_context)
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
        self.content: str = None

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, InactiveToken)
        self.content = token.field_content
        return start + 1

    def render(self, context: Any) -> str:
        return self.content
