from typing import Literal
from dataclasses import dataclass


@dataclass
class ReNode:
    pass


@dataclass
class GroundNode:
    content: str


@dataclass
class ReItem(ReNode):
    content: ReNode
    min_: int
    max_: int | Literal['float("inf")']


@dataclass
class ReAlt(ReNode):
    content: list[ReNode]


@dataclass
class ReCat(ReNode):
    content: list[ReNode]


class ENFA:

    class Context:

        def __init__(self):
            self.dict: dict[tuple[int, str], set[int]] = {}
            self.last_state = 0

        def get_new_state(self) -> int:
            self.last_state += 1
            return self.last_state - 1

    def __init__(self, start_state: int, end_state: int, states: range, dict_: dict[tuple[int, str], set[int]]):
        assert states.start == 0
        self.states = states
        self.alphabet = {s for _, s in dict_.keys()}.difference({''})
        self.start_state = start_state
        self.end_state = end_state
        self.dict = dict_

    @classmethod
    def parse(cls, node: ReNode) -> 'ENFA':
        context = cls.Context()
        start, end = cls._parse_any(node, context)
        return cls(start, end, range(context.last_state), context.dict)

    @classmethod
    def _parse_alt(cls, node: ReAlt, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        for item in node.content:
            start, end = cls._parse_any(item, context)
            context.dict[(my_start_state, '')] = {start}
            context.dict[(end, '')] = {my_end_state}
        return my_start_state, my_end_state

    @classmethod
    def _parse_cat(cls, node: ReCat, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for item in node.content:
            start, end = cls._parse_any(item, context)
            context.dict[(prev_end, '')] = {start}
            prev_end = end
        context.dict[(prev_end, '')] = {my_end_state}
        return my_start_state, my_end_state

    @classmethod
    def _parse_item(cls, node: ReItem, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for i in range(node.min_):
            start, end = cls._parse_any(node.content, context)
            context.dict[(prev_end, '')] = {start}
            prev_end = end
        if node.max_ != float('inf'):
            for i in range(node.min_, node.max_):
                start, end = cls._parse_any(node.content, context)
                context.dict[(prev_end, '')] = {start, my_end_state}
                prev_end = end
        else:
            start, end = cls._parse_any(node.content, context)
            context.dict[(prev_end, '')] = {start, my_end_state}
            context.dict[(end, '')] = {start, my_end_state}
        return my_start_state, my_end_state

    @classmethod
    def _parse_ground(cls, node: GroundNode, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        context.dict[(my_start_state, node.content)] = {my_end_state}
        return my_start_state, my_end_state

    @classmethod
    def _parse_any(cls, item: ReNode, context: Context) -> tuple[int, int]:
        if isinstance(item, ReAlt):
            return cls._parse_alt(item, context)
        elif isinstance(item, ReCat):
            return cls._parse_cat(item, context)
        elif isinstance(item, ReItem):
            return cls._parse_item(item, context)
        elif isinstance(item, GroundNode):
            return cls._parse_ground(item, context)
        else:
            raise ValueError(f'Unknown type {type(item)}')


class NFA:

    @classmethod
    def from_enfa(cls, enfa: ENFA) -> 'NFA':
        new_dict: dict[tuple[int, str], set[int]] = {}

        adj_matrix = cls._compute_epsilon_closure(enfa)
        for state in enfa.states:
            for letter in enfa.alphabet:
                edom_state = [s for s in enfa.states if adj_matrix[s][state]]  # TODO: finish

    @classmethod
    def _compute_epsilon_closure(cls, enfa: ENFA) -> list[list[bool]]:
        adj_matrix = [[False] * len(enfa.states) for _ in enfa.states]
        for i in enfa.states:
            adj_matrix[i][i] = True

        for (v, s), w in enfa.dict.items():
            if s == '':
                adj_matrix[v][w] = True

        for k in enfa.states:
            for i in enfa.states:
                for j in enfa.states:
                    adj_matrix[i][j] = adj_matrix[i][j] or (adj_matrix[i][k] and adj_matrix[k][j])

        return adj_matrix
