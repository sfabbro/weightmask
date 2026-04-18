import re

with open("tests/simulate_and_test.py", "r") as f:
    content = f.read()

# E203 whitespace before ':' fix
content = re.sub(r" \:", ":", content)

with open("tests/simulate_and_test.py", "w") as f:
    f.write(content)
