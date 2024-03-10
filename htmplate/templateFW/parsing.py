"""
parsing - tools for parsing and rendering templates

This module provides tools and interfaces for creating a custom template language.
To make a custom template language, all you need is to implement a Lexer and
any number of ControlField or SingleField classes. Then you can pass your new
implementations to Parser class; and it is done.

Classes that inherit from TreeNode and FieldSignature represent non-terminating symbols in the grammar:
non-terminals: (ContentNode, LeafNode, Field, InitialSignature(n), MiddleSignature(n), FinalSignature(n), liminal)
terminals: text derived from InactiveToken
initial: ContentNode
rules:
    ContentNode -> Field ContentNode | LeafNode ContentNode | epsilon
    LeafNode -> plain-text
    Field -> InitialSignature(n) ContentNode liminal | ContentNode
    liminal -> MiddleSignature(n) ContentNode liminal | FinalSignature(n)

The signatures are to be defined in the Field classes.
XSignature(n) represents any member of a set that contains this signature.

Terminology:

    - Field: A field is a part of the template that stores an instruction for the parser.
    - Field Signature: A field signature is a regex wrapper that matches a field instruction.
"""

from __future__ import annotations

from typing import Any, Callable
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


# LEXER =========================================================

@dataclass
class TokenBase:
    """Base class for all tokens"""
    start: int
    end: int
    field_content: str

    def __repr__(self):
        return f'<{self.__class__.__name__} field_content={self.field_content!r}>'


@dataclass
class ActiveToken(TokenBase):
    """Token that contains information on a Field"""
    instruction: str


@dataclass
class InactiveToken(TokenBase):
    """Token that contains information on a non-field part of the template"""
    pass


class LexerBase(ABC):
    """Interface for all lexers"""

    class LexerError(Exception):
        pass

    @abstractmethod
    def tokenize(self, template: str) -> list[TokenBase]:
        """
        Tokenize a template.

        This method should return a list of tokens. Each token should contain
        information on the field content, start and end position of the token
        in the template as well as the instruction if the token is an ActiveToken.

        :param template: The template to tokenize
        :return: A list of tokens"""
        pass


class SimpleLexer(LexerBase):
    """A simple lexer that uses delimiters to tokenize a template"""

    def __init__(self, delimiters: list[tuple[str, str]] = None):
        if delimiters is None:
            delimiters = [('{{', '}}')]
        self.delimiters = delimiters

    def tokenize(self, template: str | int | float | bool) -> list[TokenBase]:
        left_delimiters = '|'.join(re.escape(x[0]) for x in self.delimiters)
        right_delimiters = '|'.join(re.escape(x[1]) for x in self.delimiters)
        tag_reg = re.compile(rf'({left_delimiters})(.*?)({right_delimiters})')

        tokens = []
        prev_end = 0

        if isinstance(template, (int, float, bool)):
            return [InactiveToken(field_content=str(template), start=0, end=len(str(template)))]

        for mach in tag_reg.finditer(template):
            if prev_end != mach.start():
                tokens.append(InactiveToken(
                    field_content=template[prev_end:mach.start()],
                    start=prev_end,
                    end=mach.start()))
            tokens.append(ActiveToken(
                instruction=mach.group(2),
                field_content=template[mach.start():mach.end()],
                start=mach.start(),
                end=mach.end()))
            prev_end = mach.end()

        if prev_end != len(template):
            tokens.append(InactiveToken(field_content=template[prev_end:], start=prev_end, end=len(template)))
        return tokens


# PARSER ========================================================

class TreeNodeFactory(ABC):
    """Base class for all nodes in the parse tree"""

    @abstractmethod
    def make_tree(self,
                  start: int,
                  parser: Parser,
                  tokens: list[TokenBase],
                  fields: list[FieldFactory],
                  extra_context: Any = None
                  ) -> tuple[TreeNode, int]:
        pass


class TreeNode(ABC):

    @abstractmethod
    @property
    def factory(self) -> TreeNodeFactory:
        pass

    @abstractmethod
    def render(self, context: Any, inner_extra: dict) -> str:
        """
        Using previously parsed tree, render the template

        :param context: The context to render the template with
        :param inner_extra: store additional data (like file paths, etc.)

        :return: The rendered template
        """
        pass

    @abstractmethod
    def get_original(self) -> str:
        """:return: The original unaltered template"""
        pass

    @abstractmethod
    def to_json(self) -> Any:
        """:return: A json representation of the tree"""
        pass

    def to_json_str(self) -> str:
        """:return: a json representation of the tree as a string"""
        return json.dumps(self.to_json(), indent=4)


class FieldSignature(ABC):
    """Base class for all field signatures"""

    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, FieldSignature):
            return False
        return self.name == other.name

    @abstractmethod
    def is_a_match(self, text: str) -> bool:
        pass

    @abstractmethod
    def construct(self, text: str, context: Any, extra_context: Any, inner_extra: Any) -> dict:
        pass


class _BodyStateMachine:
    """State machine for checking the order of fields in the body of a control field"""

    def __init__(self, body: list[tuple[str, tuple[int, int]]]):
        self.body = body
        self.index = None
        self.current_occurrences = None
        self.is_started = False

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
            raise Parser.ParsingError(f'Defined body for {self.__class__.__name__} is non-deterministic'
                                      f' (Potential fields after {name}: {names})')
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


class FieldFactory(TreeNodeFactory):
    """Base class for all fields"""

    render_func_type = Callable[['Field', Any, dict], str]

    def __init__(self):
        self.signatures: list[FieldSignature] = []

        self.render_func: FieldFactory.render_func_type | None = None
        self._body: list[str, tuple[int, int]] | None = None

        self._initial_signatures: list[FieldSignature] | None = None
        self._final_signatures: list[FieldSignature] | None = None

    def add_signature(self, signature: FieldSignature):
        self.signatures.append(signature)

    def signature(self, name: str) -> Callable[[type[FieldSignature]], type[FieldSignature]]:
        def wrapper(signature_t: type[FieldSignature]) -> type[FieldSignature]:
            signature = signature_t(name)
            self.add_signature(signature)
            return signature
        return wrapper

    def body(self, *fields: tuple[str, tuple[int, int] | int] | tuple[str]):
        if self._body is not None:
            raise RuntimeError('Body already set')
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
        self._body = out

    def set_initial_signatures(self, *signatures: FieldSignature):
        if self._initial_signatures is not None:
            raise RuntimeError('Initial signatures already set')
        self._initial_signatures = signatures

    def set_final_signatures(self, *signatures: FieldSignature):
        if self._final_signatures is not None:
            raise RuntimeError('Final signatures already set')
        self._final_signatures = signatures

    @property
    def initial_signatures(self) -> list[FieldSignature]:
        return self._initial_signatures

    @property
    def middle_signatures(self) -> list[FieldSignature]:
        return [f for f in self.signatures if f not in self.initial_signatures and f not in self.final_signatures]

    @property
    def final_signatures(self) -> list[FieldSignature]:
        return self._final_signatures

    def get_matching_signature(self, instruction: str) -> FieldSignature | None:
        """
        Get the signature that matches the instruction

        :param instruction: instruction to match
        :return: if a signature is found, return it; otherwise return None
        """
        fields = [f for f in self.signatures if f.is_a_match(instruction)]
        if len(fields) > 1:
            raise Parser.ParsingError(f'Many fields found for "{instruction}" (Potential fields: {fields})')
        elif len(fields) == 1:
            return fields[0]
        else:
            return None

    def render(self, func: FieldFactory.render_func_type):
        if self.render_func is not None:
            raise RuntimeError('Render function already set')
        self.render_func = func

    def make_tree(self,
                  start: int,
                  parser: Parser,
                  tokens: list[TokenBase],
                  fields: list[FieldFactory],
                  extra_context: Any = None
                  ) -> tuple[TreeNode, int]:
        index = start
        machine = _BodyStateMachine(self._body)
        machine.start()
        out: list[tuple[FieldSignature, ActiveToken] | ContentNode] = []

        while index < len(tokens):
            current_token = tokens[index]

            if isinstance(current_token, InactiveToken):
                if not machine.next('...'):
                    raise Parser.ParsingError(f'Unexpected token "{current_token}"')
                node, index = parser.content_factory.make_tree(index, parser, tokens, fields, extra_context)
                out.append(node)

            else:
                assert isinstance(current_token, ActiveToken)
                match = self.get_matching_signature(current_token.instruction)
                if match is None:
                    raise Parser.ParsingError(f'Field for instruction "{current_token.instruction}" does not belong to '
                                              f'"{self.__class__.__name__}"')
                if not machine.next(match.name):
                    raise Parser.ParsingError(f'Unexpected field "{current_token.instruction}" '
                                              f'for "{self.__class__.__name__}"')
                out.append((match, current_token))
                if match in self.final_signatures:
                    return Field(self, out), index + 1

        raise Parser.ParsingError(f'End of template reached before body of "{self.__class__.__name__}" was closed')


class Field(TreeNode):

    def __init__(self, factory: FieldFactory, content: list[tuple[FieldSignature, ActiveToken] | ContentNode]):
        self._factory = factory
        self.content = content

    @property
    def factory(self) -> FieldFactory:
        return self._factory

    def get_original(self) -> str:
        out: list[str] = []
        for signature, token, node in self.content:
            out.append(token.field_content)
            if node is not None:
                out.append(node.get_original())
        return ''.join(out)

    def render(self, context: Any, inner_extra: dict) -> str:
        return self.factory.render_func(self, context, inner_extra)

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


class ContentNodeFactory(TreeNodeFactory):

    @staticmethod
    def _get_field(instruction: str, fields: list[FieldFactory]) -> tuple[FieldFactory, FieldSignature]:
        found = [ft for ft in fields if ft.get_matching_signature(instruction) is not None]
        if len(found) > 1:
            raise Parser.ParsingError(f'More than one field found for "{instruction}" (Found fields: {found})')
        elif len(found) == 1:
            field = found[0]
            sig = field.get_matching_signature(instruction)
            return field, sig
        else:
            raise Parser.ParsingError(f'No field found for "{instruction}"')

    def make_tree(self,
                  start: int,
                  parser: Parser,
                  tokens: list[TokenBase],
                  fields: list[FieldFactory],
                  extra_context: Any = None
                  ) -> tuple[TreeNode, int]:
        out: list[Field | LeafNode | ContentNode] = []

        current = start
        while current < len(tokens):
            token = tokens[current]

            if isinstance(token, ActiveToken):
                field, sig = self._get_field(token.instruction, fields)
                if sig in field.initial_signatures:
                    current = field.make_tree(current, parser, tokens, fields, extra_context)
                    out.append(field)
                else:
                    return ContentNode(self, out), current

            elif isinstance(token, InactiveToken):
                leaf_node, current = parser.leaf_factory.make_tree(current, parser, tokens, fields, extra_context)
                out.append(leaf_node)

            else:
                raise TypeError(f'Unknown token type {type(token)}')

        node = ContentNode(self, out)
        return node, current


class ContentNode(TreeNode):

    def __init__(self, factory: ContentNodeFactory, content: list[Field | LeafNode | ContentNode]):
        self._factory = factory
        self.content = content

    @property
    def factory(self) -> ContentNodeFactory:
        return self._factory

    def to_json(self) -> Any:
        out = []
        for child in self.content:
            out.append(child.to_json())
        return {"type": self.__class__.__name__, "content": out}

    def get_original(self) -> str:
        out: list[str] = []
        for child in self.content:
            out.append(child.get_original())
        return ''.join(out)

    def render(self, context: Any, inner_extra: dict) -> str:
        out: list[str] = []
        for child in self.content:
            out.append(child.render(context, inner_extra))
        return ''.join(out)


class LeafNodeFactory(TreeNodeFactory):

    def make_tree(self,
                  start: int,
                  parser: Parser,
                  tokens: list[TokenBase],
                  fields: list[FieldFactory],
                  extra_context: Any = None
                  ) -> tuple[LeafNode, int]:
        token = tokens[start]
        assert isinstance(token, InactiveToken)
        node = LeafNode(self, token)
        return node, start + 1


class LeafNode(TreeNode):

    def __init__(self, factory: LeafNodeFactory, content: InactiveToken):
        self._factory = factory
        self.content = content

    @property
    def factory(self) -> LeafNodeFactory:
        return self._factory

    def to_json(self) -> Any:
        return {"type": self.__class__.__name__, "content": repr(self.content)}

    def get_original(self) -> str:
        return self.content.field_content

    def render(self, context: Any, inner_extra: dict) -> str:
        return self.content.field_content


class Parser:
    """
    The parser class. This is the main class of the module

    To use this class, define custom lexer and fields and pass them to the constructor.
    You can also pass a custom ContentNode and LeafNode classes to the constructor.
    """

    class ParsingError(Exception):
        pass

    def __init__(self,
                 lexer: LexerBase,
                 fields: list[FieldFactory],
                 content_factory: ContentNodeFactory = ContentNodeFactory(),
                 leaf_factory: LeafNodeFactory = LeafNodeFactory()):
        self.content_factory = content_factory
        self.leaf_factory = leaf_factory
        self.lexer = lexer
        self.fields = fields

    def make_tree(self, template: str, extra_context: Any = None) -> ContentNode:
        """
        Make a parse tree from a template

        :param template: template to parse
        :param extra_context: extra context to pass to the nodes
        :return: abstract syntax tree
        """
        tokens = self.lexer.tokenize(template)
        root, index = self.content_factory.make_tree(0, self, tokens, self.fields, extra_context)
        if index != len(tokens):
            raise Parser.ParsingError(f'Unexpected token "{tokens[index]}";'
                                      f' parsing has ended prematurely')
        return root

    def parse(self, template: str, context: Any, **extra_context) -> str:
        """
        Parse a template.

        :param template: template to parse
        :param context: context to render the template with
        :param extra_context: extra context to pass to the nodes
        :return: parsed template
        """
        tree = self.make_tree(template, extra_context)
        return tree.render(context, {})
