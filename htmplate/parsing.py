from __future__ import annotations

from enum import Enum
from typing import Any
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class TokenBase:
    start: int
    end: int
    field_content: str

    def __repr__(self):
        return f'<{self.__class__.__name__} field_content={self.field_content!r}>'


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

    @abstractmethod
    def get_original(self) -> str:
        pass

    @abstractmethod
    def to_json(self) -> Any:
        pass

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=4)


class FieldSignature(ABC):

    def __init__(self, signature: str):
        self.signature = signature

    def match(self, text: str) -> bool:
        regex = re.compile(self.signature)
        return regex.fullmatch(text) is not None


class ControlFieldSignature(FieldSignature):

    class FieldType(Enum):
        INITIAL = 0
        MIDDLE = 1
        FINAL = 2

    def __init__(self, name: str, f_type: FieldType, signature: str, occurrences: tuple[int, int] = (1, 1)):
        super().__init__(signature)
        self.name = name
        self.f_type = f_type
        self.occurrences = occurrences

    def __repr__(self):
        return f'<{self.__class__.__name__} type={self.f_type} signature={self.signature!r}>'


class SingleFieldSignature(FieldSignature):

    def __repr__(self):
        return f'<{self.__class__.__name__} signature={self.signature!r}>'


class Field(TreeNode):

    @classmethod
    @abstractmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        pass


class ControlField(Field):

    class BodyStateMachine:

        def __init__(self, body: list[tuple[str, tuple[int, int]]]):
            self.body = body
            self.index = None
            self.current_occurrences = None
            self.is_started = False
            self.name_to_body = {b[0]: b for b in body}
            self.name_to_index = {b[0]: i for i, b in enumerate(body)}

            states = {i: {i + 1} for i in range(0, len(body) - 1)}
            states[len(body) - 1] = set()

            for i, b in reversed(list(enumerate(body))):
                tmp = frozenset(states[i])
                for x in tmp:
                    if body[x][1][0] == 0:
                        states[i].update(states[x])

            self.states = states
            self.initial_state = body[0][0]

        def get_next_name(self, state: set[int], name: str) -> int | None:
            names = [self.body[i][0] for i in state]
            if len(names) != len(set(names)):
                raise Parser.ParsingError(f'Field {name} is ambiguous. (Potential fields: {names} contain duplicates)')
            return next((i for i in state if self.body[i][0] == name), None)

        def start(self):
            self.index = -1
            self.current_occurrences = 0
            self.is_started = True

        def next(self, name: str) -> bool:
            if not self.is_started:
                raise RuntimeError('State machine not started')

            if self.index == -1:
                self.index = 0
                self.current_occurrences = 1
                return name == self.initial_state

            if self.index >= len(self.body):
                return False

            if self.body[self.index][0] == name and self.current_occurrences < self.body[self.index][1][1]:
                self.current_occurrences += 1
                return True

            else:
                lower, upper = self.body[self.index][1]
                if not (lower <= self.current_occurrences <= upper):
                    return False
                tmp = self.get_next_name(self.states[self.index], name)
                if tmp is None:
                    return False
                self.index = tmp
                self.current_occurrences = 1
                lower, upper = self.body[self.index][1]
                return self.current_occurrences <= upper

    initial_fields: list[ControlFieldSignature] = None
    middle_fields: list[ControlFieldSignature] = None
    final_fields: list[ControlFieldSignature] = None
    body: list[tuple[str, tuple[int, int]]] = None

    @staticmethod
    def make_body(*fields: tuple[str, tuple[int, int] | int] | tuple[str]) -> list[tuple[str, tuple[int, int]]]:
        out = []
        for field in fields:
            if len(field) == 1:
                name, = field
                occurrences = (1, 1)
            else:
                name, occurrences = field
            if isinstance(occurrences, int):
                occurrences = (occurrences, occurrences)
            out.append((name, occurrences))
        return out

    @staticmethod
    def initial(*fields: tuple[str, str]) -> list[ControlFieldSignature]:
        return list(map(lambda x: ControlFieldSignature(x[0], ControlFieldSignature.FieldType.INITIAL, x[1]), fields))

    @staticmethod
    def middle(*fields: tuple[str, str]) -> list[ControlFieldSignature]:
        return list(map(lambda x: ControlFieldSignature(x[0], ControlFieldSignature.FieldType.MIDDLE, x[1]), fields))

    @staticmethod
    def final(*fields: tuple[str, str]) -> list[ControlFieldSignature]:
        return list(map(lambda x: ControlFieldSignature(x[0], ControlFieldSignature.FieldType.FINAL, x[1]), fields))

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: list[tuple[ControlFieldSignature, ActiveToken, ContentNode]] = None

    def to_json(self) -> Any:
        out = []
        for signature, token, node in self.content:
            current = {
                'signature': repr(signature),
                'token': repr(token),
                'content': node.to_json() if node is not None else None
            }
            out.append(current)
        return {"type": self.__class__.__name__, "content": out}

    @classmethod
    def all_fields(cls) -> list[ControlFieldSignature]:
        return cls.initial_fields + cls.middle_fields + cls.final_fields

    @classmethod
    def get_matching_signature(cls, instruction: str) -> ControlFieldSignature | None:
        for field in cls.all_fields():
            if field.match(instruction):
                return field
        return None

    def make_tree(self, start: int) -> int:
        index = start
        machine = self.BodyStateMachine(self.body)
        machine.start()
        out: list[tuple[ControlFieldSignature, ActiveToken, ContentNode]] = []

        while index < len(self.tokens):
            current_token = self.tokens[index]
            assert isinstance(current_token, ActiveToken)

            match = self.get_matching_signature(current_token.instruction)
            if match is None:
                raise Parser.ParsingError(f'No field found for {current_token.instruction}')

            if not machine.next(match.name):
                raise Parser.ParsingError(f'Unexpected field {current_token.instruction}')

            if match.f_type in [ControlFieldSignature.FieldType.INITIAL, ControlFieldSignature.FieldType.MIDDLE]:
                content = ContentNode(self.parser, self.tokens, self.fields_t, self.extra_context)
                index = content.make_tree(index + 1)
                out.append((match, current_token, content))
            else:
                out.append((match, current_token, None))
                self.content = out
                return index + 1

        raise Parser.ParsingError('Unexpected end of template')

    def get_original(self) -> str:
        out: list[str] = []
        for signature, token, node in self.content:
            out.append(token.field_content)
            if node is not None:
                out.append(node.get_original())
        return ''.join(out)

    @abstractmethod
    def render(self, context: Any) -> str:
        pass


class SingleField(Field):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: ActiveToken | None = None

    def to_json(self) -> Any:
        return {"type": self.__class__.__name__, "content": repr(self.content)}

    @classmethod
    @abstractmethod
    def get_regex(cls) -> str:
        ...

    @classmethod
    def get_field(cls) -> ControlFieldSignature:
        return SingleFieldSignature(cls.get_regex())

    @classmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        if cls.get_field().match(instruction):
            return cls.get_field()
        return None

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, ActiveToken)
        if not self.get_field().match(token.instruction):
            raise Parser.ParsingError(f'No field found for {token.instruction}')
        self.content = token
        return start + 1

    def get_original(self) -> str:
        return self.content.field_content

    @abstractmethod
    def render(self, context: Any) -> str:
        pass


class ContentNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.children: list[Field | LeafNode | ContentNode] = None

    def to_json(self) -> Any:
        out = []
        for child in self.children:
            out.append(child.to_json())
        return {"type": self.__class__.__name__, "content": out}

    def _get_field(self, instruction: str) -> tuple[type[Field], ControlFieldSignature]:
        for field_t in self.fields_t:
            tmp = field_t.get_matching_signature(instruction)
            if tmp is not None:
                return field_t, tmp
        raise Parser.ParsingError(f'No field found for {instruction}')

    def make_tree(self, start: int) -> int:
        out: list[Field | LeafNode | ContentNode] = []

        current = start
        while current < len(self.tokens):
            token = self.tokens[current]

            if isinstance(token, ActiveToken):
                field_t, sig = self._get_field(token.instruction)
                if isinstance(sig, ControlFieldSignature):
                    if sig.f_type in [ControlFieldSignature.FieldType.MIDDLE, ControlFieldSignature.FieldType.FINAL]:
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

    def get_original(self) -> str:
        out: list[str] = []
        for child in self.children:
            out.append(child.get_original())
        return ''.join(out)

    def render(self, context: Any) -> str:
        out: list[str] = []
        for child in self.children:
            out.append(child.render(context))
        return ''.join(out)


class LeafNode(TreeNode):

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: InactiveToken = None

    def to_json(self) -> Any:
        return {"type": self.__class__.__name__, "content": repr(self.content)}

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, InactiveToken)
        self.content = token
        return start + 1

    def get_original(self) -> str:
        return self.content.field_content

    def render(self, context: Any) -> str:
        return self.content.field_content
