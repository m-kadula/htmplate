from __future__ import annotations

from typing import Any, Self
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class TokenBase:
    start: int
    end: int


@dataclass
class ActiveToken(TokenBase):
    instruction: str


@dataclass
class InactiveToken(TokenBase):
    field_content: str


class LexerBase(ABC):

    class LexerError(Exception):
        pass

    def __init__(self, left_delimiter: str, right_delimiter: str):
        self.left_delimiter = left_delimiter
        self.right_delimiter = right_delimiter

    @abstractmethod
    def tokenize(self, template: str) -> list[TokenBase]:
        pass


class SimpleLexer(LexerBase):

    def tokenize(self, template: str) -> list[TokenBase]:
        tokens = []
        prev_end = 0
        tag_reg = re.compile(self.left_delimiter + r'(.+?)' + self.right_delimiter)

        for mach in tag_reg.finditer(template):
            if prev_end != mach.start():
                tokens.append(InactiveToken(
                    field_content=template[prev_end:mach.start()],
                    start=prev_end,
                    end=mach.start()))
            tokens.append(ActiveToken(instruction=mach.group(1), start=mach.start(), end=mach.end()))
            prev_end = mach.end()

        if prev_end != len(template):
            tokens.append(InactiveToken(field_content=template[prev_end:], start=prev_end, end=len(template)))
        return tokens


class ParsingError(Exception):
    pass


class Field(ABC):

    class FieldInitError(ParsingError):
        pass

    class FieldRenderError(ParsingError):
        pass

    @abstractmethod
    def __init__(self, field_content: str, start: int, parser: Parser):
        if not self.match(field_content):
            raise self.FieldInitError(
                f'Field content {field_content} does not match field type {self.__class__.__name__}')
        self._field_content = field_content
        self._start = start
        self._parser = parser
        self._factory = parser.factory

    @property
    def parser(self) -> Parser:
        return self._parser

    @property
    def factory(self) -> ParserFactory:
        return self._factory

    @property
    def start(self) -> int:
        return self._start

    @property
    def field_content(self) -> str:
        return self._field_content

    @classmethod
    def init(cls, field_content: str, start: int, parser: Parser) -> Self:
        cls(field_content, start, parser)

    @classmethod
    @abstractmethod
    def match(cls, text: str) -> bool:
        pass

    @abstractmethod
    def render(self, context: Any, **extra_context) -> tuple[str, int]:
        pass


class TrapField:

    def __init__(self, regex: str):
        self.regex = regex
        self._compiled = re.compile(regex)

    def is_trap_field(self, field: str) -> bool:
        return self._compiled.fullmatch(field) is not None


class ParserFactory:

    def __init__(self, lexer: LexerBase, fields: list[type[Field]]):
        self.lexer = lexer
        self.fields = fields

    def get_parser(self, text: str) -> Parser:
        return Parser(self, self.lexer.tokenize(text), self.fields)

    def parse(self, text: str, context: Any, **extra_context) -> str:  # TODO: check if correct
        parser = Parser(self, self.lexer.tokenize(text), self.fields)
        return parser.parse_until(0, [], context, **extra_context)[0]


class Parser:

    def __init__(self, factory: ParserFactory, tokens: list[TokenBase], field_types: list[type[Field]]):
        self.factory = factory
        self.tokens = tokens
        self.field_types = field_types

    def _get_field(self, token: ActiveToken) -> type[Field]:
        lst = [field for field in self.field_types if field.match(token.instruction)]
        if len(lst) == 0:
            raise ParsingError(f'No field type matched {token.instruction}')
        elif len(lst) == 1:
            return lst[0]
        else:
            raise ParsingError(f'Multiple field types matched {token.instruction}')

    @staticmethod
    def _get_trap_field(regex: str, trap_fields: list[TrapField]) -> TrapField | None:
        lst = [trap_field for trap_field in trap_fields if trap_field.is_trap_field(regex)]
        if len(lst) == 0:
            return None
        elif len(lst) == 1:
            return lst[0]
        else:
            raise ParsingError(f'Multiple trap fields matched {regex}')

    def parse_until(self, start: int, trap_fields: list[TrapField],
                    context: Any, **extra_context) -> tuple[str, int, TrapField]:
        text = []
        i = start
        while i < len(self.tokens):
            token = self.tokens[i]

            if isinstance(token, InactiveToken):
                text.append(token.field_content)
                i += 1

            elif isinstance(token, ActiveToken):

                trap = self._get_trap_field(token.instruction, trap_fields)
                if trap is not None:
                    return ''.join(text), i, trap

                field_t = self._get_field(token)
                field = field_t.init(token.instruction, i, self)
                rendered, end = field.render(context, **extra_context)
                text.append(rendered)
                i = end
                continue

            else:
                raise RuntimeError(f'Unknown token type {type(token)}')

        return ''.join(text), i, None
