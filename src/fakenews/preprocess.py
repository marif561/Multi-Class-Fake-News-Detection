"""Preprocessing from the paper: remove unwanted characters and stop words."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# A small German list so Task 3b test articles are handled the same way.
GERMAN_STOP_WORDS = frozenset("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes
auch auf aus bei bin bis bist da damit dann das dass dem den denn der des dich die dies diese
diesem diesen dieser dieses dir doch dort du durch ein eine einem einen einer eines er es etwas
euch euer für gegen gewesen hab habe haben hat hatte hier hin hinter ich ihm ihn ihnen ihr ihre
im in indem ins ist jede jedem jeden jeder jedes jetzt kann kein keine man manche mein meine mich
mir mit muss nach nicht nichts noch nun nur ob oder ohne sehr sein seine sich sie sind so solche
soll sondern um und uns unser unter viel vom von vor war waren warst was weil welche wenn werde
werden wie wieder will wir wird wo zu zum zur zwar zwischen über
""".split())

STOP_WORDS = {"en": ENGLISH_STOP_WORDS, "de": GERMAN_STOP_WORDS}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
UNWANTED_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿß\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, lang: str = "en", remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text.lower())
    text = UNWANTED_RE.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        stops = STOP_WORDS.get(lang, frozenset())
        tokens = [t for t in tokens if t not in stops]
    return WS_RE.sub(" ", " ".join(tokens)).strip()
