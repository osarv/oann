# The reference llm/bpe is held to: Hugging Face's tokenizers library encoding a fixed set of texts (English, German,
# Swedish, Chinese, Japanese, Korean, Russian, code, numbers, emoji, odd whitespace) with each model's tokenizer.json,
# no special tokens added; and the chat template (transformers' apply_chat_template) of one user turn "Hi", alone and
# after the system prompt "SYS". Writes tokref.json and chatref.json beside the tokenizer.json of every model directory
# named (chatref.json holds today's date: written again on the day the test runs).
#     python3 llm/tools/tokref.py $OANN_MODELS/Qwen--Qwen3-0.6B ...        (pip install tokenizers transformers)
import json, os, sys
from tokenizers import Tokenizer

texts = [
    "Hello world! I'm here, you'll see; they've gone. WE'LL GO",
    "The year 2026 had 365 days and 1234567 people paid $3.50 each (12.5%).",
    "Über die Brücke gehen wir. Rökgräs på sjön, åäö ÅÄÖ.",
    "中文分词测试，看看效果如何。日本語のテキストです。한국어 문장입니다.",
    "Привет, как дела? Всё хорошо.",
    "def f(x):\n    return x**2  # square\n\n\n\tif a != b:\n        pass\n",
    "emoji 🎉🚀 and   spaces   \n  \r\n trailing   ",
    "<|im_start|>user\nWhat is 7*8?<|im_end|>\n<think>\n\n</think>",
    "a  12b 3.14159 v2.0.1 x86_64 1e-6",
]
for d in sys.argv[1:]:
    t = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    out = [{"text": s, "ids": t.encode(s, add_special_tokens=False).ids} for s in texts]
    json.dump(out, open(os.path.join(d, "tokref.json"), "w"), ensure_ascii=False)
    print(d, sum(len(o["ids"]) for o in out))
    from transformers import AutoTokenizer
    at = AutoTokenizer.from_pretrained(d)
    if at.chat_template:
        turns = [[{"role": "user", "content": "Hi"}], [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Hi"}]]
        json.dump([at.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in turns],
                  open(os.path.join(d, "chatref.json"), "w"), ensure_ascii=False)
