import unittest

import htmplate.nfa as automata


class ENFATest(unittest.TestCase):

    def test_alt(self):
        exp = automata.ReAlt(content=[
            automata.ReGroundNode('a'),
            automata.ReGroundNode('b'),
            automata.ReGroundNode('c')
        ])
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {2, 4, 6},
            (2, 'a'): {3},
            (3, automata.NonDeterministicAutomata.Epsilon): {1},
            (4, 'b'): {5},
            (5, automata.NonDeterministicAutomata.Epsilon): {1},
            (6, 'c'): {7},
            (7, automata.NonDeterministicAutomata.Epsilon): {1},
        }, enfa.dict)
        self.assertEqual(0, enfa.start_state)
        self.assertEqual({1}, enfa.end_states)
        self.assertEqual(set('abc'), enfa.alphabet)
        self.assertEqual(set(range(8)), enfa.states)

    def test_cat(self):
        exp = automata.ReCat(content=[
            automata.ReGroundNode('a'),
            automata.ReGroundNode('b'),
            automata.ReGroundNode('c')
        ])
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {2},
            (2, 'a'): {3},
            (3, automata.NonDeterministicAutomata.Epsilon): {4},
            (4, 'b'): {5},
            (5, automata.NonDeterministicAutomata.Epsilon): {6},
            (6, 'c'): {7},
            (7, automata.NonDeterministicAutomata.Epsilon): {1}
        }, enfa.dict)

    def test_item(self):
        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=2, max_=4)
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {2},
            (2, 'a'): {3},
            (3, automata.NonDeterministicAutomata.Epsilon): {4},
            (4, 'a'): {5},
            (5, automata.NonDeterministicAutomata.Epsilon): {1, 6},
            (6, 'a'): {7},
            (7, automata.NonDeterministicAutomata.Epsilon): {8, 1},
            (8, 'a'): {9},
            (9, automata.NonDeterministicAutomata.Epsilon): {1}
        }, enfa.dict)

        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=1, max_=1)
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {2},
            (2, 'a'): {3},
            (3, automata.NonDeterministicAutomata.Epsilon): {1}
        }, enfa.dict)

        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=0, max_=float('inf'))
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {1, 2},
            (2, 'a'): {3},
            (3, automata.NonDeterministicAutomata.Epsilon): {1, 2}
        }, enfa.dict)

    def test_recursion(self):
        exp = automata.ReCat(content=[
            automata.ReAlt(content=[
                automata.ReGroundNode('a'),
                automata.ReGroundNode('b')
            ]),
            automata.ReItem(content=automata.ReGroundNode('c'), min_=1, max_=2)
        ])
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, automata.NonDeterministicAutomata.Epsilon): {2},
            (2, automata.NonDeterministicAutomata.Epsilon): {4, 6},
            (3, automata.NonDeterministicAutomata.Epsilon): {8},
            (4, 'a'): {5},
            (5, automata.NonDeterministicAutomata.Epsilon): {3},
            (6, 'b'): {7},
            (7, automata.NonDeterministicAutomata.Epsilon): {3},
            (8, automata.NonDeterministicAutomata.Epsilon): {10},
            (9, automata.NonDeterministicAutomata.Epsilon): {1},
            (10, 'c'): {11},
            (11, automata.NonDeterministicAutomata.Epsilon): {9, 12},
            (12, 'c'): {13},
            (13, automata.NonDeterministicAutomata.Epsilon): {9}
        }, enfa.dict)

    def test_normalised(self):
        a = automata.ENFA(0, {5}, {0, 1, 2, 3, 4, 5}, {'a'}, {})
        self.assertTrue(a.is_normalised)
        b = automata.ENFA(
            5,
            {100},
            {5, 10, 100, 1000, 10000},
            {'a'},
            {(5, 'a'): {100, 10}, (1000, 'a'): {5, 10000}})
        self.assertFalse(b.is_normalised)
        b.normalise()
        self.assertTrue(b.is_normalised)


class NFATest(unittest.TestCase):

    def test_alt(self):
        exp = automata.ReAlt(content=[
            automata.ReGroundNode('a'),
            automata.ReGroundNode('b'),
            automata.ReGroundNode('c')
        ])
        nfa = automata.get_automata(exp)
        self.assertTrue(nfa.match('a'))
        self.assertTrue(nfa.match('b'))
        self.assertTrue(nfa.match('c'))
        self.assertFalse(nfa.match('d'))
        self.assertFalse(nfa.match('aa'))

    def test_if_statement(self):
        exp = automata.ReCat(content=[
            automata.ReGroundNode('if'),
            automata.ReGroundNode(...),
            automata.ReItem(content=automata.ReCat([
                automata.ReGroundNode('elif'),
                automata.ReGroundNode(...)
            ]), min_=0, max_=float('inf')),
            automata.ReItem(content=automata.ReCat([
                automata.ReGroundNode('else'),
                automata.ReGroundNode(...)
            ]), min_=0, max_=1),
            automata.ReGroundNode('endif')
        ])
        nfa = automata.get_automata(exp)
        self.assertTrue(nfa.match(['if', ..., 'elif', ..., 'elif', ..., 'else', ..., 'endif']))
        self.assertTrue(nfa.match(['if', ..., 'endif']))
        self.assertTrue(nfa.match(['if', ..., 'else', ..., 'endif']))
        self.assertTrue(nfa.match(['if', ..., 'elif', ..., 'elif', ..., 'elif', ..., 'endif']))

        self.assertFalse(nfa.match(['if', ..., 'elif', 'elif', ..., 'else', ..., 'endif']))
        self.assertFalse(nfa.match(['if', ..., 'elif', ..., 'elif', ..., 'else', ...]))
        self.assertFalse(nfa.match(['if']))


if __name__ == '__main__':
    unittest.main()
