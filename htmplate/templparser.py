from pathlib import Path
from typing import Any
import re

from .parsing import Parser, SingleField, ControlField, SimpleLexer, ActiveToken, InactiveToken


def parse_context_path(context: dict, path: str) -> Any | None:
    path = path.split('.')
    for p in path:
        if isinstance(context, list):
            if not p.isnumeric():
                return None
            else:
                context = context[int(p)]
        else:
            if p not in context:
                return None
            else:
                context = context[p]
    return context


class DataField(SingleField):

    field = SingleField.make_field(r'\s*([0-9a-zA-Z_\.]+)(?:\s+with\s+([0-9a-zA-Z_\.]+))?\s*')

    def render(self, context: Any, inner_extra: dict) -> str:
        assert isinstance(self.content, ActiveToken)
        reg = re.compile(self.field.signature)
        match = reg.match(self.content.instruction)
        name = match.group(1)

        inner = parse_context_path(context, name)
        if inner is None:
            return self.content.field_content

        if match.group(2) is not None:
            context_next = parse_context_path(context, match.group(2))
            if context_next is not None:
                context = context_next
            else:
                raise ValueError(f'Context path {match.group(2)} does not exist')

        return self.parser.parse(inner, context)


class FileInclude(SingleField):

    field = SingleField.make_field(r'\s*include\s+([0-9a-zA-Z_/\.~-]+)\s+(with\s+([0-9a-zA-Z_\.]+))?\s*')

    def render(self, context: Any, inner_extra: dict) -> str:
        assert isinstance(self.content, ActiveToken)
        assert 'fs_root' in self.extra_context

        fs_root = Path(self.extra_context['fs_root'])
        reg = re.compile(self.field.signature)
        match = reg.match(self.content.instruction)
        path = match.group(1)
        file = (fs_root / path).resolve()

        if match.group(2) is not None:
            name = match.group(3)
            inner = parse_context_path(context, name)
            if inner is not None:
                context = inner
            else:
                return self.content.field_content

        if not file.exists():
            return self.content.field_content
        with open(file, encoding='utf-8') as f:
            content = f.read()
        return self.parser.parse(content, context, **self.extra_context)


class IterField(ControlField):

    initial_fields = ControlField.initial(
        ('for', r'\s*for\s+([0-9a-zA-Z_]+)\s+in\s+([0-9a-zA-Z_\.]+)\s*')
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

        iterable = parse_context_path(context, iterable_name)
        if iterable is None:
            return self.get_original()

        if not isinstance(iterable, list):
            return self.get_original()

        for value in iterable:
            mut_context = {value_name: value}
            content = node.render(mut_context | context, inner_extra)
            out.append(content)

        tmp = ''.join(out)
        return self.parser.parse(tmp, context, **self.extra_context)


class DictIterField(ControlField):

    initial_fields = ControlField.initial(
        ('for', r'\s*for\s+([0-9a-zA-Z_]+),\s*([0-9a-zA-Z_]+)\s+in\s+([0-9a-zA-Z_\.]+)\s*')
    )

    middle_fields = ControlField.middle()

    final_fields = ControlField.final(
        ('end for', r'\s*end\s+dict\s+for\s*')
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
        key_name = match.group(1)
        value_name = match.group(2)
        iterable_name = match.group(3)

        iterable = parse_context_path(context, iterable_name)
        if iterable is None:
            return self.get_original()

        if not isinstance(iterable, dict):
            return self.get_original()

        for key, value in iterable.items():
            mut_context = {key_name: key, value_name: value}
            content = node.render(mut_context | context, inner_extra)
            out.append(content)

        tmp = ''.join(out)
        return self.parser.parse(tmp, context, **self.extra_context)


class ConditionalField(ControlField):

    initial_fields = ControlField.initial(
        ('if', r'\s*if\s+([0-9a-zA-Z_\.]+)\s*'),
        ('if', r'\s*if:exists\s+([0-9a-zA-Z_\.]+)\s*')
    )

    middle_fields = ControlField.middle(
        ('elif', r'\s*elif\s+([0-9a-zA-Z_\.]+)\s*'),
        ('elif', r'\s*elif:exists\s+([0-9a-zA-Z_\.]+)\s*'),
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
        signature = None
        node = None

        for signature, token, node in self.content:
            if signature.name in ['else', 'end if']:
                break

            check_sig = self.initial_fields[1] if signature.name == 'if' else self.middle_fields[1]

            reg = re.compile(signature.signature)
            match = reg.match(token.instruction)
            name = match.group(1)
            value = parse_context_path(context, name)

            if signature == check_sig:
                if value is not None:
                    tmp = node.render(context, inner_extra)
                    return self.parser.parse(tmp, context, **self.extra_context)
            else:
                if value is None:
                    return self.get_original()
                if bool(value):
                    tmp = node.render(context, inner_extra)
                    return self.parser.parse(tmp, context, **self.extra_context)

        if signature is not None and signature.name == 'else':
            assert node is not None
            tmp = node.render(context, inner_extra)
            return self.parser.parse(tmp, context, **self.extra_context)

        return ''


class ContextField(ControlField):

    initial_fields = ControlField.initial(
        ('context', r'\s*context\s+([0-9a-zA-Z_\.]+)\s*'),
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
        value = parse_context_path(context, value_name)
        if value is not None:
            tmp = node.render(value, inner_extra)
            return self.parser.parse(tmp, context, **self.extra_context)
        else:
            return self.get_original()
