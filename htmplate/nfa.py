from typing import Literal
from dataclasses import dataclass
from functools import reduce


@dataclass
class ReNode:
    pass


@dataclass
class GroundNode(ReNode):
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

        def add_transition(self, state: int, letter: str, s: set[int]):
            if (state, letter) not in self.dict:
                self.dict[(state, letter)] = set()
            self.dict[(state, letter)].update(s)

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
            context.add_transition(my_start_state, '', {start})
            context.add_transition(end, '', {my_end_state})
        return my_start_state, my_end_state

    @classmethod
    def _parse_cat(cls, node: ReCat, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for item in node.content:
            start, end = cls._parse_any(item, context)
            context.add_transition(prev_end, '', {start})
            prev_end = end
        context.add_transition(prev_end, '', {my_end_state})
        return my_start_state, my_end_state

    @classmethod
    def _parse_item(cls, node: ReItem, context: Context) -> tuple[int, int]:
        assert node.max_ >= node.min_ and node.max_ > 0
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for i in range(node.min_):
            start, end = cls._parse_any(node.content, context)
            context.add_transition(prev_end, '', {start})
            prev_end = end
        if node.max_ != float('inf'):
            for i in range(node.min_, node.max_):
                start, end = cls._parse_any(node.content, context)
                context.add_transition(prev_end, '', {start, my_end_state})
                prev_end = end
            context.add_transition(prev_end, '', {my_end_state})
        else:
            start, end = cls._parse_any(node.content, context)
            context.add_transition(prev_end, '', {start, my_end_state})
            context.add_transition(end, '', {start, my_end_state})
        return my_start_state, my_end_state

    @classmethod
    def _parse_ground(cls, node: GroundNode, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        context.add_transition(my_start_state, node.content, {my_end_state})
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

    def __init__(self, start_state: int, end_states: set[int], states: range, dict_: dict[tuple[int, str], set[int]]):
        assert states.start == 0
        self.states = states
        self.alphabet = {s for _, s in dict_.keys()}.difference({''})
        self.start_state = start_state
        self.end_states = end_states
        self.dict = dict_

    @classmethod
    def from_enfa(cls, enfa: ENFA) -> 'NFA':
        e_closures = cls._compute_epsilon_closure(enfa)

        new_dict: dict[tuple[int, str], set[int]] = {}
        for state in enfa.states:
            for letter in enfa.alphabet:
                transitioned = reduce(
                    lambda s1, s2: s1.union(s2),
                    (enfa.dict[(s, letter)] for s in e_closures[state] if (s, letter) in enfa.dict),
                    set()
                )
                new_dict[(state, letter)] = set()
                for s in transitioned:
                    new_dict[(state, letter)].update(e_closures[s])
                if len(new_dict[(state, letter)]) == 0:
                    del new_dict[(state, letter)]

        # TODO: delete unnecessary states

        new_end_states: set[int] = set()
        for state in enfa.states:
            if enfa.end_state in e_closures[state]:
                new_end_states.add(state)

        return cls(enfa.start_state, new_end_states, enfa.states, new_dict)

    @classmethod
    def _compute_epsilon_closure(cls, enfa: ENFA) -> dict[int, set[int]]:
        adj_matrix = [[False] * len(enfa.states) for _ in enfa.states]
        for i in enfa.states:
            adj_matrix[i][i] = True

        for (v, s), w_set in enfa.dict.items():
            for w in w_set:
                if s == '':
                    adj_matrix[v][w] = True

        for k in enfa.states:
            for i in enfa.states:
                for j in enfa.states:
                    adj_matrix[i][j] = adj_matrix[i][j] or (adj_matrix[i][k] and adj_matrix[k][j])

        e_closures: dict[int, set[int]] = dict.fromkeys(enfa.states)
        for state in enfa.states:
            e_closures[state] = set(i for i in enfa.states if adj_matrix[state][i])

        return e_closures


class NFAIterator:

    def __init__(self, nfa: 'NFA'):
        self.nfa = nfa
        self.states: frozenset[int] = frozenset({nfa.start_state})

    def next(self, letter: str):
        next_states = set()
        for state in self.states:
            next_states.update(self.nfa.dict.get((state, letter), set()))
        self.states = frozenset(next_states)

    def is_stuck(self) -> bool:
        return len(self.states) != 0  # only works if unnecessary states are eliminated

    def is_finished(self) -> bool:
        return len(self.nfa.end_states.union(self.states)) != 0
