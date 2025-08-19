# Debug preview — rag

**Question**: Quelles métriques principales et secondaires ont été utilisées ?  
Top‑K: 8  
alpha_lexical=0.35, mmr_lambda=0.6, use_rrf=True  

## Top K chunks
### #1 | score=0.8287 | len=824
La tâche associée est la reconnaissance d'images , spécifiquement de feuilles associées à 18 espèces végétales (fruits, légumineuses, céréales), dont les contours puis les détails sont progressivement analysées à travers les couches du réseau de neurones modélisé. Métriques de performance utilisées pour comparer les modèles A ce stade nous avons choisi 2 métriques. Métrique principale Accuracy : Compte tenu de la limitation du déséquilibre entre classes obtenu après la phase de pre-processing, nous avons choisi de retenir l'Accuracy comme métrique principale, définie comme le rapport entre le 

### #2 | score=0.8058 | len=263
Plusieurs traitements sont appliqués aux images dans le but d'améliorer leur qualité. Les traitements incluent des ajustements sur la luminosité, le contraste, la netteté (flou et netteté), ainsi que des redimensionnements. Les étapes sont détaillées ci-dessous :

### #3 | score=0.8141 | len=338
- Les classes minoritaires (Healthy, Bacterial blight, Brown streak disease, Green mottle) ont été massivement augmentées. - L'objectif était d'atteindre un volume comparable à la classe majoritaire (Cassava Mosaic Disease). - Les techniques utilisées incluent : rotations, flips, changements de luminosité, contrastes, translations, etc.

### #4 | score=0.8040 | len=334
L'application sera développée sur Streamlit conformément au schéma de l'application présentée en Page 6 en veillant à des règles d'ergonomie de « base » (éviter les images de fond, contraste des couleurs pour faciliter la lecture, usage des couleurs limité à une gamme réduite, boutons de navigation explicites, etc.) Jalons du projet

### #5 | score=0.8037 | len=252
La deuxième phase a été consacrée à l'interprétation des modèles en utilisant des outils comme GradCam ou SHAP ainsi qu'à l'apprentissage, par lequel nous commençons cette phase, d'un modèle alternatif aux CNN et Transformer Standard : Swintransformer.

### #6 | score=0.8166 | len=609
Nous n'avons pas eu recours à des experts externes à proprement parler (en-dehors de Damien évidemment !) mais avons réalisé de nombreuses recherches afin de nous guider, notamment sur les sujets suivants : - Taille des images standard utilisée dans les algorithmes de reconnaissance d'image ; - Détermination du niveau de flou d'une image et seuil d'acceptabilité pour la reconnaissance d'images (150 de Laplacienne) ; - Retraitement des niveaux de flous ; - Déséquilibre acceptable entre classes sur le fichier d'entraînement (1 à 10) ; Nombre minimal d'images à soumettre à l'algorithme par classe

### #7 | score=0.8037 | len=688
, Résultats Global rank = . , models = EffNetV2M Swin. , accuracy = 0.9865. , fl score = 0.9856. , nb_models = . , cassava_only = False. , Résultats Global rank = . , models = EffNetV2M ResNet5oV2. , accuracy = 0.9855. , fl score = 0.9847. , nb_models = . , cassava_only = False. , Résultats Global rank = . , models = EffNetV2M. , accuracy = 0.9858. , fl score = 0.9845. , nb_models = . , cassava_only = False. , Résultats Global rank = . , models = EffNetV2M. , accuracy = 0.9841. , fl score = 0.9837. , nb_models = . , cassava_only = False. , Résultats Global rank = . , models = EffNetV2M Swin. ,

### #8 | score=0.8032 | len=817
Width Width Avant retraitement, les images avaient des dimensions très variées, ce qui compliquait leur analyse et leur traitement uniforme. Cette hétérogénéité des tailles pouvait entraîner des biais lors de l'application de modèles d'apprentissage automatique ou d'autres analyses d'images, car les réseaux de neurones et les algorithmes de traitement d'images nécessitent des entrées de taille uniforme. Après retraitement, toutes les images ont été redimensionnées à une taille standard de 256x256 pixels, un format couramment utilisé dans la datascience. Cette normalisation permet non seulement

