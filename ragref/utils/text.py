from __future__ import annotations
import re
from typing import List, Set

_STOP = {
    "fr": {
        "le","la","les","un","une","des","de","du","au","aux","à","en","et","ou","où","dans","sur","par",
        "pour","avec","sans","ce","cet","cette","ces","qui","que","quoi","dont","est","sont","été","être",
        "sera","seront","d","l","se","sa","son","ses","leur","leurs","nos","notre","vos","votre","mes","mon","ma",
        "nous","vous","ils","elles","il","elle","on","y","ne","pas","plus","moins","comme","afin","ainsi","donc",
        "car","mais","si","quand","très","peu","bien","bon","mauvais"
    },
    "en": {
        "the","a","an","and","or","of","to","in","on","for","with","without","is","are","was","were","be",
        "been","being","this","that","these","those","who","whom","which","what","when","where","why","how",
        "it","its","his","her","their","our","your","i","you","he","she","they","we","not","no","yes","very"
    }
}
_STOPALL: Set[str] = _STOP["fr"] | _STOP["en"]
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def tok(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t]

def keywords(q: str) -> List[str]:
    return [t for t in tok(q) if len(t) >= 3 and t not in _STOPALL]