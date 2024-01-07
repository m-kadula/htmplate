import re
import unittest

from htmplate.parsing import *


class SimpleLexerTest(unittest.TestCase):

    def test_basic(self):
        text = "Hi, my name is {{name}}. and I am from {{country}}. I am {{age}} years old."
        lexer = SimpleLexer('{{', '}}')
        tokens = lexer.tokenize(text)
        it = lambda x, y, z: InactiveToken(field_content=x, start=y, end=z)
        at = lambda x, t, y, z: ActiveToken(instruction=x, field_content=t, start=y, end=z)
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
        lexer = SimpleLexer('{{', '}}')
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [])

    def test_no_tags(self):
        text = "Hi, my name is John. and I am from USA. I am 20 years old."
        lexer = SimpleLexer('{{', '}}')
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [InactiveToken(field_content=text, start=0, end=len(text))])

    def test_only_tags(self):
        text = "{{name}}{{country}}{{age}}"
        lexer = SimpleLexer('{{', '}}')
        tokens = lexer.tokenize(text)
        self.assertEqual(tokens, [ActiveToken(instruction='name', field_content='{{name}}', start=0, end=8),
                                  ActiveToken(instruction='country', field_content='{{country}}', start=8, end=19),
                                  ActiveToken(instruction='age', field_content='{{age}}', start=19, end=26)])


class DataField(Field):
    regex = r'^\s([a-zA-Z0-9_]+?)\s*$'

    def __init__(self, instruction: str, start: int, parser: Parser):
        super().__init__(instruction, start, parser)

    @classmethod
    def match(cls, text: str) -> bool:
        regex = re.compile(cls.regex)
        return regex.match(text) is not None

    def render(self, context: Any, **extra_context) -> tuple[str, int]:
        regex = re.compile(self.regex)
        data = regex.match(self.instruction).group(1)
        if data in context:
            inner_parsed = self.factory.parse(context[data], context, **extra_context)
            return inner_parsed, self.start + 1
        else:
            tmp = self.parser.tokens[self.start]
            assert isinstance(tmp, ActiveToken)
            return tmp.field_content, self.start + 1


class IterField(Field):
    regex = r'^\s*for ([a-zA-Z0-9_]+) in ([a-zA-Z0-9_]+)\s*$'
    trap = TrapField(r'^\s*end for\s*$')

    def __init__(self, instruction: str, start: int, parser: Parser):
        super().__init__(instruction, start, parser)

    @classmethod
    def match(cls, text: str) -> bool:
        regex = re.compile(cls.regex)
        return regex.match(text) is not None

    def find_end(self) -> int:
        stack = []
        for i in range(self.start + 1, len(self.parser.tokens)):
            tmp = self.parser.tokens[i]
            assert isinstance(tmp, (InactiveToken, ActiveToken))
            if isinstance(tmp, ActiveToken) and self.trap.is_trap_field(tmp.instruction):
                if len(stack) == 0:
                    return i + 1
                else:
                    stack.pop()
            elif isinstance(tmp, ActiveToken) and self.match(tmp.instruction):
                stack.append(tmp)
        else:
            raise self.FieldRenderError(f'No end for found for {self.instruction}')

    def render(self, context: Any, **extra_context) -> tuple[str, int]:
        regex = re.compile(self.regex)
        match = regex.match(self.instruction)
        variable = match.group(1)
        data = match.group(2)
        end = self.find_end()

        if data not in context:
            text = ''.join([token.field_content for token in self.parser.tokens[self.start:end]])
            return text, end

        lst = context[data]
        if not isinstance(lst, list):
            raise self.FieldRenderError(f'{data} is not a list')

        text = []
        for item in lst:
            inner_context = {**context, variable: item}
            inner_parsed, _, trap = self.parser.parse_until(self.start + 1, [self.trap], inner_context, **extra_context)
            if trap is None:
                raise self.FieldRenderError(f'No end for found for {self.instruction}')
            text.append(inner_parsed)

        return ''.join(text), end


class ParsingTest(unittest.TestCase):

    def setUp(self):
        self.factory = ParserFactory(SimpleLexer('{{', '}}'), [DataField, IterField])

    def test_basic(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'country': 'USA', 'age': '20'}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA. I am 20 years old.')

    def test_nested(self):
        text = "Hi, my name is {{ name }}. and I am from {{ nested }}. I am {{ age }} years old."
        context = {'name': 'John', 'nested': "{{ country }}, {{ town }}", 'age': '20', 'country': 'USA', 'town': 'NY'}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA, NY. I am 20 years old.')

    def test_no_val_in_context(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'age': '20'}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from {{ country }}. I am 20 years old.')

    def test_iter(self):
        text = "{{ for item in items }}item: {{ item }}\n{{ end for }}"
        context = {'items': ['a', 'b', 'c']}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\nitem: c\n')

    def test_iter_nested(self):
        text = "{{ for list in lists }}{{ for item in list }}item: {{ item }}\n{{ end for }}\n{{ end for }}"
        context = {'lists': [['a', 'b'], ['c', 'd']]}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\n\nitem: c\nitem: d\n\n')

    def test_empty_iter(self):
        text = "{{ for item in items }}item: {{ item }}\n{{ end for }}"
        context = {'items': []}
        parsed = self.factory.parse(text, context)
        self.assertEqual(parsed, '')


if __name__ == '__main__':
    unittest.main()
