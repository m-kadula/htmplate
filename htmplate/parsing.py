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

from enum import Enum
from typing import Any
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

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        self.parser = parser
        self.tokens = tokens
        self.fields_t = fields_t
        self.extra_context = extra_context

    @abstractmethod
    def make_tree(self, start: int) -> tuple[TreeNode2, int]:
        """
        Make a tree from the tokens. This tree should be a recursive structure stored
        inside implementing class.

        :param start: index of the first token to parse
        :return: the next token to parse
        """
        pass


class TreeNode2(ABC):

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

    @abstractmethod
    def is_a_match(self, text: str) -> bool:
        pass

    @abstractmethod
    def construct(self, text: str) -> dict:
        pass


class Field(TreeNode):
    """Base class for all fields"""

    @classmethod
    @abstractmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        """
        Get the signature that matches the instruction

        :param instruction: instruction to match
        :return: if a signature is found, return it; otherwise return None
        """
        pass


class ControlField(Field):
    """
    Base class for all control fields

    To implement a control field, you need to implement the following attributes:

        - initial_fields = ControlField.initial(...): list of ControlFieldSignature for initial fields
        - middle_fields = ControlField.middle(...): list of ControlFieldSignature for middle fields
        - final_fields = ControlField.final(...): list of ControlFieldSignature for final fields
        - body = ControlField.make_body(...): list of tuples of (name, occurrences) for the body of the field

        All of these attributes are class attributes and should be initialised using the static methods
        initial, middle, final and make_body.

        Initial fields are fields that start a control field. For example, in a for loop,
        the 'for' field is an initial field. In this instance a 'end for' field is a final field.
        When the parser encounters a final field, it will stop parsing the body of the control field.
        Middle field are fields that are neither initial nor final.

        The body defines the structure of the control field.
        It is a list of tuples of (name, occurrences). The name is the name of the field
        previously defined in the initial_fields, middle_fields or final_fields. The occurrences
        can be a tuple of (lower, upper) or an integer. If it is a tuple, it means that the field
        must occur at least lower times and at most upper times. If it is an integer, it means
        that the field must occur exactly that number of times. If the field is not present, it
        defaults to 1.
    """

    class BodyStateMachine:
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

    initial_fields: list[type[FieldSignature]] = None
    middle_fields: list[type[FieldSignature]] = None
    final_fields: list[type[FieldSignature]] = None
    body: list[tuple[str, tuple[int, int]]] = None

    @staticmethod
    def make_body(*fields: tuple[str, tuple[int, int] | int] | tuple[str]) -> list[tuple[str, tuple[int, int]]]:
        """
        Creates a body for a control field. This should be used to initialise the body attribute.

        :param fields: A list of tuples of (name, occurrences) or (name, (lower, upper)) or (name)
        """
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

    # @staticmethod
    # def initial(*fields: tuple[str, str]) -> list[FieldSignature]:
    #     """
    #     Creates a list of initial fields. This should be used to initialise the initial_fields attribute.
    #
    #     :param fields: a list of tuples of (name, signature) where signature is a regex that matches the field
    #     """
    #     return list(map(lambda x: ControlFieldSignature(x[0], FieldSignature.FieldType.INITIAL, x[1]), fields))
    #
    # @staticmethod
    # def middle(*fields: tuple[str, str]) -> list[ControlFieldSignature]:
    #     """
    #     Creates a list of middle fields. This should be used to initialise the middle_fields attribute.
    #     :param fields: a list of tuples of (name, signature) where signature is a regex that matches the field
    #     """
    #     return list(map(lambda x: ControlFieldSignature(x[0], ControlFieldSignature.FieldType.MIDDLE, x[1]), fields))
    #
    # @staticmethod
    # def final(*fields: tuple[str, str]) -> list[ControlFieldSignature]:
    #     """
    #     Creates a list of final fields. This should be used to initialise the final_fields attribute.
    #
    #     :param fields: a list of tuples of (name, signature) where signature is a regex that matches the field
    #     """
    #     return list(map(lambda x: ControlFieldSignature(x[0], ControlFieldSignature.FieldType.FINAL, x[1]), fields))

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: list[tuple[FieldSignature, ActiveToken, ContentNode]] = None

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
    def all_fields(cls) -> list[FieldSignature]:
        """:return: all fields (initial, middle and final)"""
        return cls.initial_fields + cls.middle_fields + cls.final_fields

    @classmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        """
        Get the signature that matches the instruction

        :param instruction: instruction to match
        :return: if a signature is found, return it; otherwise return None
        """
        fields = [f for f in cls.all_fields() if f.is_a_match(instruction)]
        if len(fields) > 1:
            raise Parser.ParsingError(f'Many fields found for "{instruction}" (Potential fields: {fields})')
        elif len(fields) == 1:
            return fields[0]
        else:
            return None

    def make_tree(self, start: int) -> int:
        index = start
        machine = self.BodyStateMachine(self.body)
        machine.start()
        out: list[tuple[FieldSignature, ActiveToken, ContentNode]] = []

        while index < len(self.tokens):
            current_token = self.tokens[index]
            assert isinstance(current_token, ActiveToken)

            match = self.get_matching_signature(current_token.instruction)
            if match is None:
                raise Parser.ParsingError(f'Field for instruction "{current_token.instruction}" does not belong to '
                                          f'"{self.__class__.__name__}"')

            if not machine.next(match.name):
                raise Parser.ParsingError(f'Unexpected field "{current_token.instruction}" '
                                          f'for "{self.__class__.__name__}"')

            if match.f_type in [ControlFieldSignature.FieldType.INITIAL, ControlFieldSignature.FieldType.MIDDLE]:
                content = self.parser.content_factory_t(self.parser, self.tokens, self.fields_t, self.extra_context)
                index = content.make_tree(index + 1)
                out.append((match, current_token, content))
            else:
                out.append((match, current_token, None))
                self.content = out
                return index + 1

        raise Parser.ParsingError(f'End of template reached before body of "{self.__class__.__name__}" was closed')

    def get_original(self) -> str:
        out: list[str] = []
        for signature, token, node in self.content:
            out.append(token.field_content)
            if node is not None:
                out.append(node.get_original())
        return ''.join(out)

    @abstractmethod
    def render(self, context: Any, inner_extra: dict) -> str:
        pass


class SingleField(Field):
    """
    Base class for all single fields

    To implement a single field, you need to implement the following attribute:
    field = SingleField.make_field(...): a SingleFieldSignature that matches the field
    This attribute is a class attribute and should be initialised using the static method make_field.

    The render method should be implemented to render the field.
    """

    field: FieldSignature = None

    @staticmethod
    def make_field(regex: str) -> SingleFieldSignature:
        """
        Creates a field signature. This should be used to initialise the field attribute.

        :param regex: regex that matches the field
        """
        return SingleFieldSignature(regex)

    def __init__(self, parser: Parser, tokens: list[TokenBase], fields_t: list[type[Field]], extra_context: Any = None):
        super().__init__(parser, tokens, fields_t, extra_context)
        self.content: ActiveToken | None = None

    def to_json(self) -> Any:
        return {"type": self.__class__.__name__, "content": repr(self.content)}

    @classmethod
    def get_field(cls) -> ControlFieldSignature:
        return cls.field

    @classmethod
    def get_matching_signature(cls, instruction: str) -> FieldSignature | None:
        """
        Get the signature that matches the instruction

        :param instruction: instruction to match
        :return: if a signature is found, return it; otherwise return None
        """
        if cls.get_field().match(instruction):
            return cls.get_field()
        return None

    def make_tree(self, start: int) -> int:
        token = self.tokens[start]
        assert isinstance(token, ActiveToken)
        if not self.get_field().match(token.instruction):
            raise Parser.ParsingError(f'Instruction "{token.instruction}" '
                                      f'does not match field {self.__class__.__name__}')
        self.content = token
        return start + 1

    def get_original(self) -> str:
        return self.content.field_content

    @abstractmethod
    def render(self, context: Any, inner_extra: dict) -> str:
        pass


class ContentNodeFactory(TreeNodeFactory):

    def _get_field(self, instruction: str) -> tuple[type[Field], FieldSignature]:
        found = [ft for ft in self.fields_t if ft.get_matching_signature(instruction) is not None]
        if len(found) > 1:
            raise Parser.ParsingError(f'More than one field found for "{instruction}" (Found fields: {found})')
        elif len(found) == 1:
            field_t = found[0]
            sig = field_t.get_matching_signature(instruction)
            return field_t, sig
        else:
            raise Parser.ParsingError(f'No field found for "{instruction}"')

    def make_tree(self, start: int) -> tuple[ContentNode, int]:
        out: list[Field | LeafNode | ContentNode] = []

        current = start
        while current < len(self.tokens):
            token = self.tokens[current]

            if isinstance(token, ActiveToken):
                field_t, sig = self._get_field(token.instruction)
                if isinstance(sig, FieldSignature):
                    if sig.f_type in [ControlFieldSignature.FieldType.MIDDLE, ControlFieldSignature.FieldType.FINAL]:
                        self.children = out
                        return current
                field = field_t(self.parser, self.tokens, self.fields_t, self.extra_context)
                current = field.make_tree(current)
                out.append(field)

            elif isinstance(token, InactiveToken):
                factory = self.parser.leaf_factory_t(self.parser, self.tokens, self.fields_t, self.extra_context)
                leaf_node, current = factory.make_tree(current)
                out.append(leaf_node)

            else:
                raise TypeError(f'Unknown token type {type(token)}')

        node = ContentNode(self, out)
        return node, current


class ContentNode(TreeNode2):

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

    def make_tree(self, start: int) -> tuple[LeafNode, int]:
        token = self.tokens[start]
        assert isinstance(token, InactiveToken)
        node = LeafNode(self, token)
        return node, start + 1


class LeafNode(TreeNode2):

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
                 field_types: list[type[ControlField]],
                 content_factory_t: type[ContentNodeFactory] = ContentNode,
                 leaf_factory_t: type[LeafNodeFactory] = LeafNode):
        self.content_factory_t = content_factory_t
        self.leaf_factory_t = leaf_factory_t
        self.lexer = lexer
        self.field_types = field_types

    def make_tree(self, template: str, extra_context: Any = None) -> ContentNode:
        """
        Make a parse tree from a template

        :param template: template to parse
        :param extra_context: extra context to pass to the nodes
        :return: abstract syntax tree
        """
        tokens = self.lexer.tokenize(template)
        root = self.content_factory_t(self, tokens, self.field_types, extra_context)
        out = root.make_tree(0)
        if out != len(tokens):
            raise Parser.ParsingError(f'Unexpected token "{tokens[out]}";'
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
