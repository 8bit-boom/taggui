from pathlib import Path

from auto_captioning.auto_captioning_model import replace_template_variables
from utils.image import Image


def make_image(path: str, tags: list[str]) -> Image:
    return Image(path=Path(path), dimensions=(100, 100), tags=tags)


def test_replace_template_variables_tags():
    image = make_image('/dataset/subdir/pic.jpg', ['cat', 'outdoors'])
    result = replace_template_variables('Tags: {tags}', image)
    assert result == 'Tags: cat, outdoors'


def test_replace_template_variables_name():
    image = make_image('/dataset/subdir/pic.jpg', [])
    result = replace_template_variables('Name: {name}', image)
    assert result == 'Name: pic'


def test_replace_template_variables_directory():
    image = make_image('/dataset/subdir/pic.jpg', [])
    result = replace_template_variables('Dir: {directory}', image)
    assert result == 'Dir: subdir'
    result = replace_template_variables('Dir: {folder}', image)
    assert result == 'Dir: subdir'


def test_replace_template_variables_case_insensitive():
    image = make_image('/dataset/subdir/pic.jpg', ['cat'])
    result = replace_template_variables('{TAGS}', image)
    assert result == 'cat'


def test_replace_template_variables_escaped_braces_are_unescaped():
    image = make_image('/dataset/subdir/pic.jpg', ['cat'])
    result = replace_template_variables(r'literal \{tags\} and {tags}', image)
    assert result == 'literal {tags} and cat'


def test_replace_template_variables_unknown_variable_is_removed():
    image = make_image('/dataset/subdir/pic.jpg', [])
    result = replace_template_variables('before {nonsense} after', image)
    assert result == 'before  after'
