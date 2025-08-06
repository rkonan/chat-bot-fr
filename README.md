---
title: Recoplante Streamlit
emoji: 🌿🔍
colorFrom: green
colorTo: yellow
sdk: docker
pinned: true
---




Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

#  installer le projet en local
git clone https://github.com/rkonan/reco-plante.git
cd reco-plante
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
lancement sur api local =>  export API_ENV=local && streamlit run app/main.py 
lancement sur api distante =>  export API_ENV=space && streamlit run app/main.py 

# 🚀 Déploiement automatique vers Hugging Face Spaces

Ce dépôt est relié automatiquement à un **Hugging Face Space** :  
🔗 [https://huggingface.co/spaces/reco-team/reco-plante](https://huggingface.co/spaces/reco-team/reco-plante)

---

## ⚙️ Workflow de déploiement

À chaque `push` sur la branche `main`, un workflow GitHub Actions est déclenché :  
- Il vérifie le code
- Il pousse automatiquement le contenu vers le Space `reco-team/reco-plante` sur Hugging Face

Les contributeurs **n'ont pas besoin d'un compte Hugging Face** :  
✅ Tout le déploiement est géré par GitHub Actions.

---

## 📦 Structure technique

- Workflow : `.github/workflows/deploy.yaml`
- Branche surveillée : `main`
- Hugging Face Space cible : `reco-team/reco-plante`

---

## 🔒 Sécurité

Le workflow utilise un **secret `HF_TOKEN` stocké dans GitHub** :  
- Ce token a des permissions `write` sur Hugging Face Spaces
- Il n'est jamais exposé dans le code source

---

## 👩‍💻 Usage pour les développeurs

- **Collaborer normalement sur GitHub**
- **Créer des PRs ou pousser directement sur `main` si autorisé**
- Aucun `git push` manuel vers Hugging Face n’est requis

---

## 📝 Remarque importante

💡 Si vous avez besoin de **tester le Space localement avant de pousser**, vous pouvez cloner le repo et tester avec :  
```bash
lancement sur api local =>  export API_ENV=local && streamlit run app/main.py 
lancement sur api distante =>  export API_ENV=space && streamlit run app/main.py 
