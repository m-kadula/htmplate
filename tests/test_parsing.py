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

    @classmethod
    def get_regex(cls) -> str:
        return r'\s*([0-9a-zA-Z_]+)\s*'

    def execute_field(self, context: Any) -> str:
        assert isinstance(self.content, ActiveToken)
        reg = re.compile(self.get_regex())
        match = reg.match(self.content.instruction)
        name = match.group(1)
        if name in context:
            inner = context[name]
            return self.parser.parse(inner, context)
        else:
            return self.content.field_content


class IterField(ControlField):

    def start_context(self) -> ControlField.Storage:
        return ControlField.Storage()

    def check_context(self, storage: ControlField.Storage, signature: ControlFieldSignature):
        pass

    def start_internal_storage(self):
        storage = ControlField.Storage()
        storage.index = 0
        storage.iterable = None
        storage.iterated = None
        return storage


@IterField.initial('for', r'\s*for\s+([0-9a-zA-Z_]+)\s+in\s+([0-9a-zA-Z_]+)\s*')
def initial(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    sig, token, content_node = field.get_elements(control)
    reg = re.compile(sig.signature)
    match = reg.match(token.instruction)
    if control.internal.iterable is None:
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


@IterField.final('end for', r'\s*end\s+for\s*')
def final(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    control.internal.index += 1
    if control.internal.index < len(control.context[control.internal.iterated]):
        control.index = 0
        return control, None
    else:
        control.exit_next = True
        return control, None


class ConditionalField(ControlField):

    def start_context(self) -> ControlField.Storage:
        storage = ControlField.Storage()
        storage.last_command = None
        return storage

    def check_context(self, storage: ControlField.Storage, signature: ControlFieldSignature) -> ControlField.Storage:
        current_command = signature.name
        if storage.last_command == current_command and not storage.last_command == 'elif':
            raise ValueError(f"Cannot have two consecutive {current_command} commands")
        if storage.last_command == 'else' and current_command != 'end if':
            raise ValueError(f"Cannot have {current_command} after else command")
        storage.last_command = current_command
        return storage

    def start_internal_storage(self):
        pass


@ConditionalField.initial('if', r'\s*if\s+([0-9a-zA-Z_]+)\s*')
def if_stmt(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    sig, token, content_node = field.get_elements(control)
    reg = re.compile(sig.signature)
    match = reg.match(token.instruction)
    value = match.group(1)
    if value in control.context:
        value = control.context[value]
        if value:
            control.exit_next = True
            return control, content_node.render(context=control.context) if value else None
        else:
            control.index += 1
            return control, None
    else:
        control.exit_next = True
        out = []
        for signature, token, node in field.content:
            tmp = node.render({}) if node is not None else ''
            out.append(token.field_content + tmp)
        return control, ''.join(out)


@ConditionalField.middle('elif', r'\s*elif\s+([0-9a-zA-Z_]+)\s*')
def elif_stmt(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    return if_stmt(field, control)


@ConditionalField.middle('else', r'\s*if:else\s*')
def else_stmt(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    sig, token, content_node = field.get_elements(control)
    control.exit_next = True
    return control, content_node.render(context=control.context)


@ConditionalField.final('end if', r'\s*end\s+if\s*')
def end_if_stmt(field: ControlField, control: ControlField.ControlFlowInfo) -> tuple[ControlField.ControlFlowInfo, str]:
    control.exit_next = True
    return control, None


class TreeTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(SimpleLexer(), [DataField, IterField, ConditionalField])

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
        self.parser = Parser(SimpleLexer(), [DataField, IterField, ConditionalField])

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


if __name__ == '__main__':
    unittest.main()
