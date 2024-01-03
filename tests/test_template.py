import json
from pathlib import Path
from unittest import TestCase

from htmplate.template import HTMLTemplate

THIS_DIR = Path(__file__).resolve().parent


class TestTemplate(TestCase):

    def setUp(self):
        self.info_path = THIS_DIR / 'info'
        self.template_path = THIS_DIR / 'templates'
        self.correct_path = THIS_DIR / 'correct'
        self.out_path = THIS_DIR / 'out_dir'

        # if self.out_path.exists():
        #     for file in os.listdir(self.out_path):
        #         os.remove(self.out_path / file)
        # else:
        #     self.out_path.mkdir()

    def test_general(self):
        with open(self.template_path / 'leenvit_footer.htm') as f:
            self.template = f.read()
        with open(self.info_path / 'people.json') as f:
            self.info = json.load(f)

        generator = HTMLTemplate(self.template)
        generated_html = generator.make(self.info['Anna Nowak'])
        self.assertTrue(generated_html.is_final)
        with open(self.correct_path / 'Anna Nowak.html') as f:
            correct_html = f.read()
        self.assertEqual(correct_html, generated_html.template_content)
