import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from htmplate.parsing import Parser, SimpleLexer
from htmplate.templparser import DataField, IterField, DictIterField, ConditionalField, ContextField, FileInclude


RESOURCE_DIR = Path(__file__).parent / 'resources'


class ParsingTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser(
            SimpleLexer(),
            [DataField, IterField, DictIterField, ConditionalField, ContextField, FileInclude])

    def test_basic(self):
        text = "Hi, my name is {{ name }}. and I am from {{ country }}. I am {{ age }} years old."
        context = {'name': 'John', 'country': 'USA', 'age': '20'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA. I am 20 years old.')

        text = "Hi, my name is {{ name }}. and I am from {{ nested }}. I am {{ age }} years old."
        context = {'name': 'John', 'nested': "{{ country }}, {{ town }}", 'age': '20', 'country': 'USA', 'town': 'NY'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA, NY. I am 20 years old.')

    def test_indexing(self):
        text = "{{ list.1.2 }} {{ list.0.name }} {{ dict.name }} {{ dict.null }} {{ list.20 }}"
        context = {
            'list': [
                {'name': 'John', 'age': '20'},
                [1, 2, 3, 4]
            ],
            'dict': {
                'name': 'John',
                'age': '20'
            }
        }
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, '3 John John {{ dict.null }} {{ list.20 }}')

    def test_types(self):
        context = {
            'int': 1,
            'float': 3.141592653589793,
            'str': 'John',
            'bool': True,
        }
        text = "{{ int }} {{ float }} {{ str }} {{ bool }}"
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, '1 3.141592653589793 John True')

    def test_data_field_with(self):
        text = "Hi, my name is {{ name }}. and I am from {{ nested with dat.info }}."
        context = {'name': 'John', 'nested': "{{ country }}, {{ town }}",
                   'dat': {'info': {'country': 'USA', 'town': 'NY'}}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Hi, my name is John. and I am from USA, NY.')

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

        text = "{{ for item in items }}item: {{ item }}\n{{ end for }}{{ item }}"
        context = {'items': ['a', 'b', 'c']}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'item: a\nitem: b\nitem: c\n{{ item }}')

    def test_dict_iter(self):
        text = "{{ for key, value in dct }}{{ key }}: {{ value }}\n{{ end dict for }}"
        context = {'dct': {'a': '1', 'b': '2', 'c': '3'}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'a: 1\nb: 2\nc: 3\n')

    def test_iter_with_file(self):
        text = ("{{ for con in lists }}{{ con.name }}:\n"
                "{{ include test_footer.html with con }}\n\n{{ end for }}")
        context = {
            'lists': [
                {
                    "name": "Walter White",
                    "contact_info": {
                        "emails": ["walter.white@gmail.com", "waltuh@gmail.com"],
                        "numbers": ["+1 123 456 7890", "+1 999 999 9999"]
                    }
                },
                {
                    "name": "Jessie Pinkman",
                    "contact_info": {
                        "emails": ["jessie.pinkman@gmail.com", "jessie@icloud.com"],
                        "numbers": ["+1 098 765 4321", "+1 111 111 1111"]
                    }
                }
            ]
        }
        parsed = self.parser.parse(text, context, fs_root=RESOURCE_DIR)
        with open(RESOURCE_DIR / 'list_of_html_filled.txt') as f:
            expected = f.read()
        soup_got = BeautifulSoup(parsed, 'html.parser')
        soup_expected = BeautifulSoup(expected, 'html.parser')
        self.assertEqual(soup_got.prettify(), soup_expected.prettify())

    def test_iter_recursion(self):
        text = "{{ for item in items }}{{ recursive }}{{ end for }}"
        context = {'items': ['a', 'b', 'c'], 'recursive': "{{name}}", 'name': 'John'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'JohnJohnJohn')

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

    def test_if_exists(self):
        text = "{{ if:exists name }}{{ name }}{{ if:else }}{{ name2 }}{{ end if }}"
        context = {'name': 'John', 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')
        context = {'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Mike')

        text = "{{ if condition }}{{ name }}{{ elif:exists name2 }}{{ name2 }}{{ end if }}"
        context = {'condition': False, 'name': 'John', 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'Mike')
        context = {'condition': False, 'name': 'John'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, '')

    def test_if_type(self):
        text = ("{{ if:typeof name str }}name: {{ name }}"
                "{{elif:typeof name list}}names:\n"
                "{{ for n in name }}\t{{ n }}\n{{ end for }}{{ end if }}")
        context = {'name': 'John'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'name: John')

        context = {'name': ['John', 'Mike', 'Jack']}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'names:\n\tJohn\n\tMike\n\tJack\n')

        text = "{{ if:typeof name dict }}dict{{ elif:exists name2 }}exists{{ if:else }}else{{ end if }}"
        context = {'name': {'a': 1}, 'name2': 'Mike'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'dict')

        context = {'name': 12, 'name2': 'a'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'exists')

        context = {'name': 'a'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'else')

    def test_context(self):
        text = "{{ context context_name }}{{ name }}{{ end context }}"
        context = {'context_name': {'name': 'John'}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'John')

        text = "{{ context null }}{{ name }}{{ end context }}"
        context = {'context_name': {'name': 'John'}}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, text)

    def test_recursive(self):
        text = "outer {{ inner }} outer_end"
        inner = "inner {{ inner2 }} inner_end"
        inner2 = "inner2 {{ inner3 }} inner2_end"
        context = {'inner': inner, 'inner2': inner2, 'inner3': 'ground'}
        parsed = self.parser.parse(text, context)
        self.assertEqual(parsed, 'outer inner inner2 ground inner2_end inner_end outer_end')

    def test_file(self):
        text = "<meta>{{ include test_footer.html }}</meta>"
        with open(RESOURCE_DIR / 'test_footer_filled.html') as f:
            expected = f.read()

        context = {
            "name": "John Doe",
            "contact_info": {
                "emails": ["john.doe@mail.com", "john.doe@gmail.com"],
                "numbers": ["123456789", "123 456 789"]
            }
        }
        parsed = self.parser.parse(text, context, fs_root=RESOURCE_DIR)
        soup_got = BeautifulSoup(parsed, 'html.parser')
        soup_expected = BeautifulSoup('<meta>' + expected + '</meta>', 'html.parser')
        self.assertEqual(soup_got.prettify(), soup_expected.prettify())

        text = "<meta>{{ include does_not_exist.html }}</meta>"
        parsed = self.parser.parse(text, context, fs_root=RESOURCE_DIR)
        self.assertEqual(parsed, text)

        text = "<meta>{{ include test_footer.html with inner.innermost }}</meta>"
        context = {
            'inner': {
                'innermost': {
                    "name": "John Doe",
                    "contact_info": {
                        "emails": ["john.doe@mail.com", "john.doe@gmail.com"],
                        "numbers": ["123456789", "123 456 789"]
                    }
                }
            }
        }

        parsed = self.parser.parse(text, context, fs_root=RESOURCE_DIR)
        soup_got = BeautifulSoup(parsed, 'html.parser')
        self.assertEqual(soup_got.prettify(), soup_expected.prettify())

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

    def test_html(self):
        with open(RESOURCE_DIR / 'test_footer.html') as f:
            text = f.read()
        with open(RESOURCE_DIR / 'test_footer_filled.html') as f:
            expected = f.read()
        context = {
            "name": "John Doe",
            "contact_info": {
                "emails": ["john.doe@mail.com", "john.doe@gmail.com"],
                "numbers": ["123456789", "123 456 789"]
            }
        }
        parsed = self.parser.parse(text, context)
        soup_got = BeautifulSoup(parsed, 'html.parser')
        soup_expected = BeautifulSoup(expected, 'html.parser')
        self.assertEqual(soup_got.prettify(), soup_expected.prettify())


if __name__ == '__main__':
    unittest.main()
