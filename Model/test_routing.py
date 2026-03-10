from intent_router import _is_non_python_language_request, route_prompt

test_cases = [
    "give me the code for finding the factorial of a number",
    "generate python function",
    "hi",
    "give me java code for factorial",
]

for text in test_cases:
    is_non_py = _is_non_python_language_request(text)
    route = route_prompt(text)
    print(f"Text: {text}")
    print(f"  Non-Python lang: {is_non_py}")
    print(f"  Route: {route}")
    print()
