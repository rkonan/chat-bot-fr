import os
import tarfile
from pathlib import Path

# Localisation du cache HF par défaut (Linux/macOS)
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

# Si tu veux forcer un autre chemin, décommente et adapte :
# cache_dir = Path("/chemin/vers/ton/cache")

# On filtre uniquement les modèles Docling
targets = ["models--ds4sd--docling-layout-old", "models--ds4sd--docling-models"]

archive_path = Path("docling_models_cache.tar.gz")

with tarfile.open(archive_path, "w:gz") as tar:
    for t in targets:
        folder = cache_dir / t
        if folder.exists():
            print(f"[OK] Ajout de {folder}")
            tar.add(folder, arcname=t)
        else:
            print(f"[WARN] {folder} introuvable, ignoré")

print(f"\n✅ Archive créée : {archive_path.resolve()}")
