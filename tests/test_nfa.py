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
            (0, ''): {2, 4, 6},
            (2, 'a'): {3},
            (3, ''): {1},
            (4, 'b'): {5},
            (5, ''): {1},
            (6, 'c'): {7},
            (7, ''): {1},
        }, enfa.dict)
        self.assertEqual(0, enfa.start_state)
        self.assertEqual(1, enfa.end_state)
        self.assertEqual(set('abc'), enfa.alphabet)

    def test_cat(self):
        exp = automata.ReCat(content=[
            automata.ReGroundNode('a'),
            automata.ReGroundNode('b'),
            automata.ReGroundNode('c')
        ])
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, ''): {2},
            (2, 'a'): {3},
            (3, ''): {4},
            (4, 'b'): {5},
            (5, ''): {6},
            (6, 'c'): {7},
            (7, ''): {1}
        }, enfa.dict)

    def test_item(self):
        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=2, max_=4)
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, ''): {2},
            (2, 'a'): {3},
            (3, ''): {4},
            (4, 'a'): {5},
            (5, ''): {1, 6},
            (6, 'a'): {7},
            (7, ''): {8, 1},
            (8, 'a'): {9},
            (9, ''): {1}
        }, enfa.dict)

        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=1, max_=1)
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, ''): {2},
            (2, 'a'): {3},
            (3, ''): {1}
        }, enfa.dict)

        exp = automata.ReItem(content=automata.ReGroundNode('a'), min_=0, max_=float('inf'))
        enfa = automata.ENFA.parse(exp)
        self.assertEqual({
            (0, ''): {1, 2},
            (2, 'a'): {3},
            (3, ''): {1, 2}
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
            (0, ''): {2},
            (2, ''): {4, 6},
            (3, ''): {8},
            (4, 'a'): {5},
            (5, ''): {3},
            (6, 'b'): {7},
            (7, ''): {3},
            (8, ''): {10},
            (9, ''): {1},
            (10, 'c'): {11},
            (11, ''): {9, 12},
            (12, 'c'): {13},
            (13, ''): {9}
        }, enfa.dict)


class NFATest(unittest.TestCase):

    def test_alt(self):
        exp = automata.ReAlt(content=[
            automata.ReGroundNode('a'),
            automata.ReGroundNode('b'),
            automata.ReGroundNode('c')
        ])
        enfa = automata.ENFA.parse(exp)
        nfa = automata.NFA.from_enfa(enfa)
        self.assertEqual({}, nfa.dict)


if __name__ == '__main__':
    unittest.main()
