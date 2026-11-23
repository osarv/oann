# The reference speech/g2p.olang is held to: espeak-ng's phonemes as the phonemizer package gives them (backend espeak,
# en-us and sv, preserve_punctuation=True, with_stress=True) - what kakao-enterprise/vits-ljs and Piper's voices were
# trained on. espeak-ng (GPL-3) is only run here, offline, never vendored or linked.
#
#   apt-get install espeak-ng; pip install phonemizer
#   python3 -I speech/tools/g2p_reference.py [OUT]          # OUT: ~/.cache/oann/ref/g2p
#
# Word lists are counted from public-domain books (Project Gutenberg, fetched from its mirror into OUT/texts): English
# from Pride and Prejudice, The Adventures of Sherlock Holmes, Tom Sawyer and Moby Dick - the 2000 most frequent words,
# and 200 more frequent ones that CMUdict lacks; Swedish from Lagerlof's Bannlyst (1918), Soderberg's Forvillelser
# (1895) and Gertrud Almqvist's I tolfte timmen - the 1000 most frequent. Each file is a line a word or sentence:
# "text<TAB>phonemes".

import collections
import os
import re
import sys
import urllib.request

from phonemizer.backend import EspeakBackend

MIRROR = "https://gutenberg.pglaf.org"
CMUDICT = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

BOOKS = {
    "en": [("1342", "1342-0.txt", "utf-8"), ("1661", "1661-0.txt", "utf-8-sig"), ("74", "74-0.txt", "utf-8-sig"),
           ("2701", "2701-0.txt", "utf-8")],
    "sv": [("39147", "39147-8.txt", "latin-1"), ("21862", "21862-8.txt", "latin-1"), ("77571", "77571-0.txt", "utf-8")],
}

SENTENCES = {
    "en": [
        "Hello, world! The quick brown fox jumps over the lazy dog.",
        "I have 23 apples and 1,000 dollars; it was 1999.",
        "She said: \"It's not what you think.\"",
        "What time is it? It's half past seven, I believe.",
        "The weather today is sunny, with a high of 75 degrees.",
        "Turn on the lights in the kitchen, please.",
        "Mr. Smith and Dr. Jones arrived at 10 o'clock.",
        "Water, butter and a little city party.",
        "Remind me to call my mother tomorrow at noon.",
        "There were 3 cats, 12 dogs and 101 birds!",
        "Is this the real life, or is this just fantasy?",
        "My phone number is 555 0123.",
    ],
    "sv": [
        "Hej, världen! Sju sjösjuka sjömän.",
        "Jag har 23 äpplen och 1000 kronor.",
        "Vad är klockan? Hon är halv åtta.",
        "Tänd lampan i köket, tack.",
        "Det blir soligt i morgon, med upp till 18 grader.",
        "Kärlek, stjärna och kyrka.",
        "Påminn mig om att ringa mamma klockan tolv.",
        "Barnen springer över gården; korten ligger på bordet.",
        "Hur mycket kostar det? Det kostar 45 kronor.",
        "Skeppet seglade genom skären i går kväll.",
        "Han sa: \"Det här är inte sant.\"",
        "Vi ses på måndag, klockan 9.",
    ],
}


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def book(out, ident, name, encoding):
    path = fetch(f"{MIRROR}/{'/'.join(ident[:-1])}/{ident}/{name}", os.path.join(out, "texts", ident + ".txt"))
    text = open(path, encoding=encoding).read()
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    return text[text.find("\n", start) if start >= 0 else 0:end if end >= 0 else len(text)]


def counts(out, lang):
    c = collections.Counter()
    for ident, name, enc in BOOKS[lang]:
        for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", book(out, ident, name, enc)):
            if w[0].isupper() and w.lower() != "i":
                c["^" + w.lower()] += 1     # capitalised: counted apart, kept only if mostly lower case
            else:
                c[w.lower()] += 1
    return c


def frequent(c, n, keep):
    words = []
    for w, k in c.most_common():
        if w.startswith("^") or c["^" + w] > k:
            continue
        if keep(w):
            words.append(w)
        if len(words) == n:
            break
    return words


def write(backend, path, texts):
    phones = backend.phonemize(texts, strip=True)
    with open(path, "w", encoding="utf-8") as f:
        for t, p in zip(texts, phones):
            f.write(f"{t}\t{p}\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/oann/ref/g2p")
    os.makedirs(os.path.join(out, "texts"), exist_ok=True)
    cmu = set()
    for line in open(fetch(CMUDICT, os.path.join(out, "texts", "cmudict.dict")), encoding="utf-8"):
        cmu.add(line.split(" ", 1)[0].split("(")[0])
    en = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
    c = counts(out, "en")
    write(en, os.path.join(out, "en_words.tsv"), frequent(c, 2000, lambda w: True))
    write(en, os.path.join(out, "en_oov.tsv"), frequent(c, 200, lambda w: w not in cmu and len(w) > 2 and "'" not in w))
    write(en, os.path.join(out, "en_sentences.tsv"), SENTENCES["en"])
    sv = EspeakBackend("sv", preserve_punctuation=True, with_stress=True)
    c = counts(out, "sv")
    write(sv, os.path.join(out, "sv_words.tsv"), frequent(c, 1000, lambda w: "qv" not in w and "'" not in w))
    write(sv, os.path.join(out, "sv_sentences.tsv"), SENTENCES["sv"])


if __name__ == "__main__":
    main()
