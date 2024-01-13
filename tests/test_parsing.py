import unittest

from htmplate.parsing import *


class SimpleLexerTest(unittest.TestCase):

    def test_basic(self):
        def it(x, y, z):
            return InactiveToken(field_content=x, start=y, end=z)

        def at(x, t, y, z):
            return ActiveToken(instruction=x, field_content=t, start=y, end=z)

        text = "Hi, my name is {{name}}. and I am from {{country}}. I am {{age}} years old."
        lexer = SimpleLexer()
        tokens = lexer.tokenize(text)
        self.assertEqual(len(tokens), 7)
        self.assertEqual([it('Hi, my name is ', 0, 15),
                          at('name', '{{name}}', 15, 23),
                          it('. and I am from ', 23, 39),
                          at('country', '{{country}}', 39, 50),
                          it('. I am ', 50, 57),
                          at('age', '{{age}}', 57, 64),
                          it(' years old.', 64, 75)],
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


class DataField(SingleField):

    field = SingleField.make_field(r'\s*([0-9a-zA-Z_]+)\s*')

    def render(self, context: Any, inner_extra: dict) -> str:
        assert isinstance(self.content, ActiveToken)
        reg = re.compile(self.field.signature)
        match = reg.match(self.content.instruction)
        name = match.group(1)
        if name in context:
            inner = context[name]
            return self.parser.parse(inner, context)
        else:
            return self.content.field_content


class IterField(ControlField):

    initial_fields = ControlField.initial(
        ('for', r'\s*for\s+([0-9a-zA-Z_]+)\s+in\s+([0-9a-zA-Z_]+)\s*')
    )

    middle_fields = ControlField.middle()

    final_fields = ControlField.final(
        ('end for', r'\s*end\s+for\s*')
    )

    body = ControlField.make_body(
        ('for',),
        ('end for',)
    )

    def render(self, context: Any, inner_extra: dict) -> str:
        out: list[str] = []
        signature, token, node = self.content[0]
        reg = re.compile(signature.signature)
        match = reg.match(token.instruction)
        value_name = match.group(1)
        iterable_name = match.group(2)

        if iterable_name not in context:
            return self.get_original()

        iterable = context[iterable_name]
        for value in iterable:
            mut_context = {value_name: value}
            content = node.render(mut_context | context, inner_extra)
            out.append(content)

        return ''.join(out)


class ConditionalField(ControlField):

    initial_fields = ControlField.initial(
        ('if', r'\s*if\s+([0-9a-zA-Z_]+)\s*'),
    )

    middle_fields = ControlField.middle(
        ('elif', r'\s*elif\s+([0-9a-zA-Z_]+)\s*'),
        ('else', r'\s*if:else\s*'),
    )

    final_fields = ControlField.final(
        ('end if', r'\s*end\s+if\s*'),
    )

    body = ControlField.make_body(
        ('if',),
        ('elif', (0, float('inf'))),
        ('else', (0, 1)),
        ('end if',)
    )

    def render(self, context: Any, inner_extra: dict) -> str:
        for signature, token, node in self.content:
            if signature.name in ('if', 'elif'):
                reg = re.compile(signature.signature)
                match = reg.match(token.instruction)
                value_name = match.group(1)
                if value_name in context:
                    value = context[value_name]
                    if value:
                        return node.render(context, inner_extra)
                else:
                    return self.get_original()
            elif signature.name == 'else':
                return node.render(context, inner_extra)
        return ''


class ContextField(ControlField):

    initial_fields = ControlField.initial(
        ('context', r'\s*context\s+([0-9a-zA-Z_]+)\s*'),
    )

    middle_fields = ControlField.middle()

    final_fields = ControlField.final(
        ('end context', r'\s*end\s+context\s*'),
    )

    body = ControlField.make_body(
        ('context',),
        ('end context',)
    )

    def render(self, context: Any, inner_extra: dict) -> str:
        signature, token, node = self.content[0]
        reg = re.compile(signature.signature)
        match = reg.match(token.instruction)
        value_name = match.group(1)
        if value_name in context:
            value = context[value_name]
            return node.render(value, inner_extra)
        else:
            return self.get_original()


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

    def test_basic(self):
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

    def test_nesting(self):
        text = "{{ for item in items }}{{ item }}{{ end for }}"
        expected = """{
    "type": "ContentNode",
    "content": [
        {
            "type": "IterField",
            "content": [
                {
                    "signature": "<ControlFieldSignature type=FieldType.INITIAL \
signature='\\\\\\\\s*for\\\\\\\\s+([0-9a-zA-Z_]+)\\\\\\\\s+in\\\\\\\\s+([0-9a-zA-Z_]+)\\\\\\\\s*'>",
                    "token": "ActiveToken(start=0, end=23, field_content='{{ for item in items }}', \
instruction=' for item in items ')",
                    "content": {
                        "type": "ContentNode",
                        "content": [
                            {
                                "type": "DataField",
                                "content": "ActiveToken(start=23, end=33, field_content='{{ item }}', \
instruction=' item ')"
                            }
                        ]
                    }
                },
                {
                    "signature": "<ControlFieldSignature type=FieldType.FINAL \
signature='\\\\\\\\s*end\\\\\\\\s+for\\\\\\\\s*'>",
                    "token": "ActiveToken(start=33, end=46, field_content='{{ end for }}', instruction=' end for ')",
                    "content": null
                }
            ]
        }
    ]
}"""
        tree = self.parser.make_tree(text)
        self.assertEqual(tree.to_json_str(), expected)


class ParsingTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(SimpleLexer(), [DataField, IterField, ConditionalField, ContextField])

    def test_basic(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'country': 'USA', 'age': '20'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA. I am 20 years old.')

        text = "Hi, my name is {{ name }}. and I am from {{ nested }}. I am {{ age }} years old."
        context = {'name': 'John', 'nested': "{{ country }}, {{ town }}", 'age': '20', 'country': 'USA', 'town': 'NY'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA, NY. I am 20 years old.')

    def test_no_val_in_context(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'age': '20'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from {{ country }}. I am 20 years old.')

    def test_no_val_in_context_if(self):
        text = "{{ if condition }}my name is: {{ name }}{{ if:else }}i wont tell you my name.{{ end if }}"
        context = {}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, text)

    def test_iter(self):
        text = "{{ for item in items }}item: {{ item }}\n{{ end for }} hi"
        context = {'items': ['a', 'b', 'c']}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\nitem: c\n hi')

        text = "{{ for list in lists }}{{ for item in list }}item: {{ item }}\n{{ end for }}\n{{ end for }}"
        context = {'lists': [['a', 'b'], ['c', 'd']]}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\n\nitem: c\nitem: d\n\n')

        text = "{{ for item in items }}item: {{ item }}\n{{ end for }}"
        context = {'items': []}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, '')

    def test_if(self):
        text = "{{ if condition }}{{ name }}{{ end if }}"
        context = {'condition': True, 'name': 'John'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')

        text = "{{ if condition }}my name is: {{ name }}{{ if:else }}i wont tell you my name.{{ end if }}"
        parsed = self.parser.parse(text, {'condition': False, 'name': 'John'})
        self.assertEqual(parsed, 'i wont tell you my name.')
        parsed = self.parser.parse(text, {'condition': True, 'name': 'John'})
        self.assertEqual(parsed, 'my name is: John')

        text = ("{{ if condition1 }}{{ name1 }}{{ elif condition2 }}"
                "{{ name2 }}{{ elif condition3 }}{{ name3 }}{{ end if }}")
        context = {'condition1': False, 'condition2': True, 'condition3': False,
                   'name1': 'John', 'name2': 'Mike', 'name3': 'Jack'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Mike')
        context = {'condition1': False, 'condition2': False, 'condition3': True,
                   'name1': 'John', 'name2': 'Mike', 'name3': 'Jack'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Jack')
        context = {'condition1': True, 'condition2': False, 'condition3': False,
                   'name1': 'John', 'name2': 'Mike', 'name3': 'Jack'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')

        text = "{{ if condition1 }}{{ name1 }}{{ elif condition2 }}{{ name2 }}{{ if:else }}nobody{{ end if }}"
        context = {'condition1': False, 'condition2': True, 'name1': 'John', 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Mike')
        context = {'condition1': False, 'condition2': False, 'name1': 'John', 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'nobody')
        context = {'condition1': True, 'condition2': False, 'name1': 'John', 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')

    def test_context(self):
        text = "{{ context context_name }}{{ name }}{{ end context }}"
        context = {'context_name': {'name': 'John'}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')

        text = "{{ context null }}{{ name }}{{ end context }}"
        context = {'context_name': {'name': 'John'}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, text)

    def test_whole(self):
        text = """
Hi my name is {{ name }}. I am {{ age }} years old.
a random list:{{ context iteration }}{{ for list in lists }}{{ for item in list }}
{{ if item }}    - {{ item }}{{ if:else }}    - empty :({{ end if }}{{ end for }}
{{ end for }}
{{ end context }}
        """
        context = {'name': 'John', 'age': '20', 'iteration': {'lists': [['one', 'two', 'three'], ['raz', 'dwa', '']]}}
        parsed = self.parser.parse(text, context)
        print(parsed)
        expected = """
Hi my name is John. I am 20 years old.
a random list:
    - one
    - two
    - three

    - raz
    - dwa
    - empty :(
"""
        self.assertEqual(expected.strip(), parsed.strip())


if __name__ == '__main__':
    unittest.main()
