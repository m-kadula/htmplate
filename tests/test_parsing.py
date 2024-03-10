import unittest

from htmplate.templateFW.parsing import *
from htmplate.templparser import DataField, ContextField, IterField, ConditionalField


class SimpleLexerTest(unittest.TestCase):

    @staticmethod
    def it(x, y, z):
        return InactiveToken(field_content=x, start=y, end=z)

    @staticmethod
    def at(x, t, y, z):
        return ActiveToken(instruction=x, field_content=t, start=y, end=z)

    def test_many_delimiters(self):
        lexer = SimpleLexer([('{{', '}}'), ('{%', '%}')])
        text = "My names are {% for name in names %}name: {{ name }}.{% end for %}."
        tmp = lexer.tokenize(text)
        self.assertEqual(tmp,
                         [
                                self.it('My names are ', 0, 13),
                                self.at(' for name in names ', '{% for name in names %}', 13, 36),
                                self.it('name: ', 36, 42),
                                self.at(' name ', '{{ name }}', 42, 52),
                                self.it('.', 52, 53),
                                self.at(' end for ', '{% end for %}', 53, 66),
                                self.it('.', 66, 67)
                         ])

    def test_basic(self):
        text = "Hi, my name is {{name}}. and I am from {{country}}. I am {{age}} years old {{}} ."
        lexer = SimpleLexer()
        tokens = lexer.tokenize(text)
        self.assertEqual(len(tokens), 9)
        self.assertEqual([self.it('Hi, my name is ', 0, 15),
                          self.at('name', '{{name}}', 15, 23),
                          self.it('. and I am from ', 23, 39),
                          self.at('country', '{{country}}', 39, 50),
                          self.it('. I am ', 50, 57),
                          self.at('age', '{{age}}', 57, 64),
                          self.it(' years old ', 64, 75),
                          self.at('', '{{}}', 75, 79),
                          self.it(' .', 79, 81)],
                         tokens)

    def test_empty(self):
        text = ""
        lexer = SimpleLexer()
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [])

    def test_no_tags(self):
        text = "Hi, my name is John. and I am from USA. I am 20 years old."
        lexer = SimpleLexer()
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [InactiveToken(field_content=text, start=0, end=len(text))])

    def test_only_tags(self):
        text = "{{name}}{{country}}{{age}}"
        lexer = SimpleLexer()
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [ActiveToken(instruction='name', field_content='{{name}}', start=0, end=8),
                                  ActiveToken(instruction='country', field_content='{{country}}', start=8, end=19),
                                  ActiveToken(instruction='age', field_content='{{age}}', start=19, end=26)])


class TreeTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(SimpleLexer(), [DataField, IterField, ConditionalField, ContextField])

    def test_state_machine(self):
        class Test(ControlField):
            initial_fields = ControlField.initial(
                ('a', r'\s*a\s*')
            )
            middle_fields = ControlField.middle(
                ('b', r'\s*b\s*'),
                ('c', r'\s*c\s*'),
            )
            final_fields = ControlField.final(
                ('d', r'\s*d\s*')
            )
            body = ControlField.make_body(
                ('a',),
                ('b', (0, 1)),
                ('c', (0, 10)),
                ('a', 1),
                ('c', (0, 1)),
                ('d',)
            )

            def render(self, context: Any, inner_extra: dict) -> str:
                return ''

        mini_tests = [
            ("a b c c c a c d", True),
            ("a a d", True),
            ("a d", False),
            ("a b c d", False),
            ("b c c c a c d", False),
            ("a b b c a c d", False),
            ("a b c c c c c c c a c d", True),
        ]

        machine = ControlField.BodyStateMachine(Test.body)
        for text, expected in mini_tests:
            machine.start()
            text = text.split()
            self.assertEqual(expected, all(machine.next(x) for x in text))

        class Test2(Test):
            body = ControlField.make_body(
                ('a',),
                ('b', (0, 1)),
                ('b', (0, 10)),
                ('d',)
            )

        text = 'a b d'
        machine = ControlField.BodyStateMachine(Test2.body)
        machine.start()
        text = text.split()
        self.assertRaises(Parser.ParsingError, lambda: all(machine.next(x) for x in text))

    def test_state_machine_same_names(self):
        class Test(ControlField):
            initial_fields = ControlField.initial(
                ('a', r'\s*a\s*'),
                ('a', r'\s*a2\s*'),
                ('a', r'\s*a3\s*')
            )
            middle_fields = ControlField.middle(
                ('b', r'\s*b\s*'),
                ('c', r'\s*c\s*'),
                ('c', r'\s*c2\s*'),
                ('x', r'\s*x\s*'),
            )
            final_fields = ControlField.final(
                ('d', r'\s*d\s*'),
                ('d', r'\s*d2\s*')
            )
            body = ControlField.make_body(
                ('a',),
                ('b', (0, 1)),
                ('c', (0, 10)),
                ('x', 1),
                ('c', (0, 1)),
                ('d',)
            )

            def render(self, context: Any, inner_extra: dict) -> str:
                out = []
                for signature, token, node in self.content:
                    out.append(signature.name)
                return ' '.join(out)

        parser = Parser(SimpleLexer(), [Test])

        mini_tests = [
            ("a b c c2 c x c d", True),
            ("a2 x d", True),
            ("a3 d", False),
            ("a x c d", True),
            ("b3 c c2 c x c d2", False),
            ("a2 b b c x c d", False),
            ("a b c c2 c c2 c c c x c d2", True),
        ]

        for text, no_error in mini_tests:
            expected = ' '.join(x[0] for x in text.split())
            text2 = ''.join('{{' + a + '}}' for a in text.split())
            if no_error:
                self.assertEqual(expected, parser.parse(text2, {}))
            else:
                self.assertRaises(Parser.ParsingError, lambda: parser.make_tree(text2))

    def test_if_exception(self):
        text = "{{ if condition }}{{ name }}{{ if:else }}{{ name2 }}{{ elif c2 }}{{ name3 }}{{ end if }}"
        context = {'condition': True, "c2": True, 'name': 'John', 'name2': 'Mike', 'name3': 'Jack'}
        self.assertRaises(Parser.ParsingError, lambda: self.parser.parse(text, context))

    def test_small_tree(self):
        text = "test {{ test }}"
        expected = """
        {
    "type": "ContentNode",
    "content": [
        {
            "type": "LeafNode",
            "content": "InactiveToken(start=0, end=5, field_content='test ')"
        },
        {
            "type": "DataField",
            "content": "ActiveToken(start=5, end=15, field_content='{{ test }}', instruction=' test ')"
        }
    ]
}""".strip()
        tree = self.parser.make_tree(text)
        self.assertEqual(tree.to_json_str(), expected)


if __name__ == '__main__':
    unittest.main()
