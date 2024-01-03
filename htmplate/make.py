import re


def make(template: str, values: dict[str, str]) -> str:
    regex = re.compile(r'%%([a-zA-Z0-9_ -]+)%%')
    html = []
    last_start = 0
    for match in re.finditer(regex, template):
        key = match.group(1)
        html.append(template[last_start:match.start()])
        html.append(values[key])
        last_start = match.end()
    html.append(template[last_start:])
    return ''.join(html)
