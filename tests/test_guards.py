from sk_guardrails.guards.regex import RegexGuard
from sk_guardrails.engine import Engine

# Create a RegexGuard (only letters and spaces allowed)
rg = RegexGuard(r"^[a-zA-Z ]+$")

# Engine with a single guard
engine = Engine([rg])

# Test inputs
tests = [
    "Hello StreamKnight",  # Should pass
    "Hello123",            # Should fail
    "Hello World!"         # Should fail due to "!"
]

for t in tests:
    result = engine.run(t)
    print(f"Input: '{t}' -> Passes: {result}")
