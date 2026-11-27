# Reference for llm/jinja: renders a fixed set of conversations with each model's own chat template through
# transformers' apply_chat_template (tokenize=False) and writes inputs and outputs to <dir>/jinjaref.json, which
# llm/chat.olang's test renders again and compares byte for byte. strftime_now writes today's date, so run it the
# day the test runs:  python3 llm/tools/chatref.py $OANN_MODELS/*/
import json
import sys

from transformers import AutoTokenizer

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city, \"live\" — in °C.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city, e.g. Göteborg"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

CASES = [
    {"name": "user", "messages": [{"role": "user", "content": "Hello there!"}], "add_generation_prompt": True},
    {"name": "system", "messages": [{"role": "system", "content": "You are terse."},
                                    {"role": "user", "content": "What is 2+2?"}], "add_generation_prompt": True},
    {"name": "multi", "messages": [{"role": "system", "content": "Be kind."},
                                   {"role": "user", "content": "Hi"},
                                   {"role": "assistant", "content": "Hello! How can I help?"},
                                   {"role": "user", "content": "Tell me a joke.\nA short one."}],
     "add_generation_prompt": True},
    {"name": "nogen", "messages": [{"role": "user", "content": "Hi"},
                                   {"role": "assistant", "content": "  Hello!  "}], "add_generation_prompt": False},
    {"name": "nothink", "messages": [{"role": "user", "content": "Hi"}], "add_generation_prompt": True,
     "extra": {"enable_thinking": False}},
    {"name": "tools", "messages": [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}}]},
        {"role": "tool", "name": "get_weather", "content": "{\"temperature\": 22}"}],
     "tools": [WEATHER], "add_generation_prompt": True},
]


def main():
    for d in sys.argv[1:]:
        d = d.rstrip("/")
        tok = AutoTokenizer.from_pretrained(d)
        out = []
        for c in CASES:
            r = dict(c)
            try:
                r["text"] = tok.apply_chat_template(c["messages"], tools=c.get("tools"), tokenize=False,
                                                    add_generation_prompt=c["add_generation_prompt"],
                                                    **c.get("extra", {}))
            except Exception as e:  # a template's raise_exception: the olang side must fail too
                r["error"] = str(e)
            out.append(r)
        with open(d + "/jinjaref.json", "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(d, [("ERR" if "error" in r else len(r["text"])) for r in out])


main()
