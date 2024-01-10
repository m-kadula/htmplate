import unittest

from htmplate.parsing import *


class SimpleLexerTest(unittest.TestCase):

    def test_basic(self):
        text = "Hi, my name is {{name}}. and I am from {{country}}. I am {{age}} years old."
        lexer = SimpleLexer()
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

    @classmethod
    def get_regex(cls) -> str:
        return r'\s*([0-9a-zA-Z_]+)\s*'

    def execute_field(self, context: Any) -> str:
        assert isinstance(self.content, ActiveToken)
        field = self.get_field()
        reg = re.compile(self.get_regex())
        match = reg.match(self.content.instruction)
        name = match.group(1)
        if name in context:
            inner = context[name]
            return self.parser.parse(inner, context)
        else:
            return self.content.field_content


class IterField(ControlField):
    def start_context(self):
        pass

    def check_context(self, signature: ControlFieldSignature):
        pass


@IterField.initial(r'\s*for\s+([0-9a-zA-Z_]+)\s+in\s+([0-9a-zA-Z_]+)\s*')
def initial(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    sig, token, content_node = field.get_elements(control)
    reg = re.compile(sig.signature)
    match = reg.match(token.instruction)
    try:
        control.internal.index
    except AttributeError:
        control.internal.index = 0
        control.internal.iterable = match.group(1)
        control.internal.iterated = match.group(2)

        if control.internal.iterated not in control.context:
            content = content_node.render(context={})
            control.exit_next = True
            return control, field.content[0][1].field_content + content + field.content[1][1].field_content

    if len(control.context[control.internal.iterated]) == 0:
        control.exit_next = True
        return control, None
    current = control.context[control.internal.iterated][control.internal.index]
    mut_context = {control.internal.iterable: current}
    content = content_node.render(mut_context | control.context)
    control.index = 1
    return control, content


@IterField.final(r'\s*end\s+for\s*')
def final(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    sig, token, content_node = field.get_elements(control)
    control.internal.index += 1
    if control.internal.index < len(control.context[control.internal.iterated]):
        control.index = 0
        return control, None
    else:
        control.exit_next = True
        return control, None


class TreeTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(SimpleLexer(), [DataField, IterField])

    def test_basic(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        tree = self.parser.make_tree(text)
        print(repr(tree))

    def test_nesting(self):
        text = ("Hi, my name is {{ name }}. I like: {{ for item in items }}{{ item }}, {{ end for }}. "
                "I am {{ age }} years old.")
        tree = self.parser.make_tree(text)
        print(repr(tree))


class ParsingTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(SimpleLexer(), [DataField, IterField])

    def test_basic(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'country': 'USA', 'age': '20'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA. I am 20 years old.')

    def test_nested(self):
        text = "Hi, my name is {{ name }}. and I am from {{ nested }}. I am {{ age }} years old."
        context = {'name': 'John', 'nested': "{{ country }}, {{ town }}", 'age': '20', 'country': 'USA', 'town': 'NY'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA, NY. I am 20 years old.')

    def test_no_val_in_context(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'age': '20'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from {{ country }}. I am 20 years old.')

    def test_iter(self):
        text = "{{ for item in items }}item: {{ item }}\n{{ end for }} hihi"
        context = {'items': ['a', 'b', 'c']}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\nitem: c\n hihi')

    def test_iter_nested(self):
        text = "{{ for list in lists }}{{ for item in list }}item: {{ item }}\n{{ end for }}\n{{ end for }}"
        context = {'lists': [['a', 'b'], ['c', 'd']]}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\n\nitem: c\nitem: d\n\n')

    def test_empty_iter(self):
        text = "{{ for item in items }}item: {{ item }}\n{{ end for }}"
        context = {'items': []}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, '')


if __name__ == '__main__':
    unittest.main()
