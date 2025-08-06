import os
import nltk

# Forcer le cache dans /tmp
NLTK_CACHE_DIR = "/tmp/nltk_data"

def patch_llamaindex_nltk():
    try:
        from llama_index.core.utils import GlobalsHelper
        class PatchedGlobalsHelper(GlobalsHelper):
            def __init__(self):
                # Rediriger vers /tmp
                self._nltk_data_dir = NLTK_CACHE_DIR
                # Télécharger punkt si nécessaire
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt", download_dir=self._nltk_data_dir)
    except Exception as e:
        print("[patch_llamaindex_nltk] Failed to patch:", e)
