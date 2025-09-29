from .base import BaseGuard
import re

class RegexGuard(BaseGuard):
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def validate(self, text):
        return bool(self.pattern.match(text))
