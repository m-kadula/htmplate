from abc import ABC, abstractmethod
from typing import Literal, Iterable, Any
from functools import reduce


def get_automata(expression: 'ReNode') -> 'NFA':
    return NFA.from_enfa_optimised(ENFA.parse(expression))


class Context:

    def __init__(self):
        self.dict: dict[tuple[int, Any], set[int]] = {}
        self.last_state = 0

    def get_new_state(self) -> int:
        self.last_state += 1
        return self.last_state - 1

    def add_transition(self, state: int, letter: Any, s: set[int]):
        if (state, letter) not in self.dict:
            self.dict[(state, letter)] = set()
        self.dict[(state, letter)].update(s)


class ReNode(ABC):

    @abstractmethod
    def parse(self, context: Context) -> tuple[int, int]:
        pass


class ReGroundNode(ReNode):

    def __init__(self, content: Any):
        assert content != NonDeterministicAutomata.Epsilon
        self.content = content

    def parse(self, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        context.add_transition(my_start_state, self.content, {my_end_state})
        return my_start_state, my_end_state


class ReItem(ReNode):

    def __init__(self, content: ReNode, min_: int, max_: int | Literal['float("inf")']):
        assert max_ >= min_ and max_ > 0
        self.content = content
        self.min_ = min_
        self.max_ = max_

    def parse(self, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for i in range(self.min_):
            start, end = self.content.parse(context)
            context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {start})
            prev_end = end
        if self.max_ != float('inf'):
            for i in range(self.min_, self.max_):
                start, end = self.content.parse(context)
                context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {start, my_end_state})
                prev_end = end
            context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {my_end_state})
        else:
            start, end = self.content.parse(context)
            context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {start, my_end_state})
            context.add_transition(end, NonDeterministicAutomata.Epsilon, {start, my_end_state})
        return my_start_state, my_end_state


class ReAlt(ReNode):

    def __init__(self, content: list[ReNode]):
        self.content = content

    def parse(self, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        for item in self.content:
            start, end = item.parse(context)
            context.add_transition(my_start_state, NonDeterministicAutomata.Epsilon, {start})
            context.add_transition(end, NonDeterministicAutomata.Epsilon, {my_end_state})
        return my_start_state, my_end_state


class ReCat(ReNode):

    def __init__(self, content: list[ReNode]):
        self.content = content

    def parse(self, context: Context) -> tuple[int, int]:
        my_start_state = context.get_new_state()
        my_end_state = context.get_new_state()
        prev_end = my_start_state
        for item in self.content:
            start, end = item.parse(context)
            context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {start})
            prev_end = end
        context.add_transition(prev_end, NonDeterministicAutomata.Epsilon, {my_end_state})
        return my_start_state, my_end_state


class NonDeterministicAutomata:

    class _EpsilonClass:
        __instance = None

        def __new__(cls):
            if cls.__instance is None:
                cls.__instance = super().__new__(cls)
            return cls.__instance

        def __eq__(self, other) -> bool:
            return other is self

        def __hash__(self):
            return hash('epsilon')

    Epsilon = _EpsilonClass()

    def __init__(self,
                 start_state: int,
                 end_states: set[int],
                 states: set[int],
                 alphabet: set[Any],
                 dict_: dict[tuple[int, str], set[int]]):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.end_states = end_states
        self.dict = dict_

    @property
    def is_normalised(self) -> bool:
        return self.states == set(range(max(self.states) + 1)) and self.start_state == 0

    def normalise(self):
        transition = (
                {self.start_state: 0}
                | {state: i for i, state in enumerate(self.states.difference({self.start_state}), start=1)}
        )
        self.start_state = 0
        self.end_states = set(transition[i] for i in self.end_states)
        self.states = set(transition.values())
        self.dict = {(transition[i], s): set(map(lambda x: transition[x], w)) for (i, s), w in self.dict.items()}


class ENFA(NonDeterministicAutomata):

    @classmethod
    def parse(cls, node: ReNode) -> 'ENFA':
        context = Context()
        start, end = node.parse(context)
        alphabet = {s for _, s in context.dict.keys()}.difference({NonDeterministicAutomata.Epsilon})
        states = set(range(context.last_state))
        return cls(start, {end}, states, alphabet, context.dict)


class NFA(NonDeterministicAutomata):

    def iterator(self) -> 'NFAIterator':
        return NFAIterator(self)

    def match(self, expression: Iterable[Any]) -> bool:
        iterator = self.iterator()
        for item in expression:
            iterator.next(item)
            if iterator.is_stuck():
                return False
        return iterator.is_finished()

    @classmethod
    def from_enfa(cls, enfa: ENFA) -> 'NFA':
        assert enfa.is_normalised

        e_closures = cls._compute_epsilon_closures(len(enfa.states), enfa.dict)

        new_dict: dict[tuple[int, Any], set[int]] = {}
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

        new_end_states: set[int] = set()
        for state in enfa.states:
            if len(enfa.end_states.intersection(e_closures[state])) != 0:
                new_end_states.add(state)

        return cls(enfa.start_state, new_end_states, enfa.states, enfa.alphabet, new_dict)

    @classmethod
    def from_enfa_optimised(cls, enfa: ENFA) -> 'NFA':
        nfa = cls.from_enfa(enfa)
        nfa.optimise()
        return nfa

    @classmethod
    def _compute_epsilon_closures(cls, states: int, dict_: dict[tuple[int, Any], set[int]]) -> dict[int, set[int]]:
        adj_matrix = [[False] * states for _ in range(states)]
        for i in range(states):
            adj_matrix[i][i] = True

        for (v, s), w_set in dict_.items():
            if s == NonDeterministicAutomata.Epsilon:
                for w in w_set:
                    adj_matrix[v][w] = True

        for k in range(states):
            for i in range(states):
                for j in range(states):
                    adj_matrix[i][j] = adj_matrix[i][j] or (adj_matrix[i][k] and adj_matrix[k][j])

        e_closures: dict[int, set[int]] = dict.fromkeys(range(states))
        for state in range(states):
            e_closures[state] = set(s for s in range(states) if adj_matrix[state][s])

        return e_closures

    def optimise(self):
        if not self.is_normalised:
            self.normalise()
        non_generating = self._get_non_generating_states()
        self._delete_states(non_generating)
        self.normalise()
        unreachable = self._get_unreachable_states()
        self._delete_states(unreachable)
        self.normalise()

    def _delete_states(self, delete_set: set[int]):
        assert self.start_state not in delete_set

        self.states.difference_update(delete_set)
        self.end_states.difference_update(delete_set)

        to_be_deleted: list[tuple[int, Any]] = []
        for (v, s), w_set in self.dict.items():
            if v in delete_set:
                to_be_deleted.append((v, s))
            else:
                w_set.difference_update(delete_set)
        for key in to_be_deleted:
            del self.dict[key]

    def _get_non_generating_states(self) -> set[int]:
        assert self.is_normalised
        states_no = len(self.states)
        adj_matrix = [[False] * states_no for _ in range(states_no)]
        for i in range(states_no):
            adj_matrix[i][i] = True

        for (v, s), w_set in self.dict.items():
            assert s != NonDeterministicAutomata.Epsilon
            for w in w_set:
                adj_matrix[v][w] = True

        for k in range(states_no):
            for i in range(states_no):
                for j in range(states_no):
                    adj_matrix[i][j] = adj_matrix[i][j] or (adj_matrix[i][k] and adj_matrix[k][j])

        non_generating = set()
        for s1 in range(states_no):
            if not any(adj_matrix[s1][s2] for s2 in self.end_states):
                non_generating.add(s1)

        return non_generating

    def _get_unreachable_states(self) -> set[int]:
        assert self.is_normalised

        adj: dict[int, set[int]] = dict.fromkeys(self.states, set())
        for (v, _), w_set in self.dict.items():
            adj[v].update(w_set)

        stack = [self.start_state]
        visited = [False] * len(self.states)
        visited[self.start_state] = True

        while stack:
            current_state = stack[-1]
            if adj[current_state]:
                next_state = adj[current_state].pop()
                if not visited[next_state]:
                    stack.append(next_state)
                    visited[next_state] = True
            else:
                stack.pop()

        return set(filter(lambda s: not visited[s], self.states))


class NFAIterator:

    def __init__(self, nfa: 'NFA'):
        self.nfa = nfa
        self.states: frozenset[int] = frozenset({nfa.start_state})

    def next(self, letter: Any):
        next_states = set()
        for state in self.states:
            next_states.update(self.nfa.dict.get((state, letter), set()))
        self.states = frozenset(next_states)

    def is_stuck(self) -> bool:
        return len(self.states) == 0  # only works if unnecessary states are eliminated

    def is_finished(self) -> bool:
        return len(self.nfa.end_states.intersection(self.states)) != 0
