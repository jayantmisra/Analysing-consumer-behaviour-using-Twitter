from cleantext import clean
import re

def remove_emojies(text):
    text = clean(text, no_emoji=True)
    return text


def url_free_text(text):
    text = re.sub(r'http\S+', 'url', text)
    return text