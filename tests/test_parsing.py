import unittest

from htmplate.parsing import *


class SimpleLexerTest(unittest.TestCase):

    def test_basic(self):
        text = "Hi, my name is {{name}}. and I am from {{country}}. I am {{age}} years old."
        lexer = SimpleLexer('{{', '}}')
        tokens = lexer.tokenize(text)
        it = lambda x, y, z: InactiveToken(field_content=x, start=y, end=z)
        at = lambda x, y, z: ActiveToken(instruction=x, start=y, end=z)
        self.assertEqual(len(tokens), 7)
        self.assertEqual([it('Hi, my name is ', 0, 15),
                          at('name', 15, 23),
                          it('. and I am from ', 23, 39),
                          at('country', 39, 50), it('. I am ', 50, 57), at('age', 57, 64),
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
        self.assertEqual(tokens, [ActiveToken(instruction='name', start=0, end=8),
                                  ActiveToken(instruction='country', start=8, end=19),
                                  ActiveToken(instruction='age', start=19, end=26)])


class ParsingTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
