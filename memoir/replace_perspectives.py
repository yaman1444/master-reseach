with open("Memoir_Master_classification_cancer_breast.md", "r") as f:
    content = f.read()

old_perspectives = """## 5.3 Perspectives : déploiement au CHU d’Abidjan et nouvelles architectures

Plusieurs pistes de travail se dégagent pour prolonger ce mémoire et aller vers un déploiement progressif dans un environnement comme le CHU d’Abidjan.

Sur le plan des données, la priorité est la constitution d’une base de données locale d’échographies mammaires annotées, idéalement multi‑centre (CHU et cliniques privées), couvrant la diversité des appareils, des protocoles d’acquisition et des profils anatomiques ivoiriens. Cette bio‑banque permettrait de réentraîner ou affiner le modèle DenseNet‑121 sur des données représentatives de la population cible, voire de constituer un dataset panafricain susceptible d’être partagé pour la recherche.

Sur le plan des architectures, plusieurs évolutions sont envisageables :

* explorer des modèles multimodaux combinant échographie et mammographie, voire des données tabulaires (âge, antécédents, facteurs de risque) dans un réseau joint (par exemple CNN \+ MLP ou Transformers multimodaux) ;  
* tester des architectures récentes plus légères et adaptées au déploiement sur matériel contraint (EfficientNet, MobileNet, Transformers compacts), tout en comparant systématiquement leur AUC et leur recall sur la classe *début* à DenseNet‑121 ;  
* intégrer des techniques avancées d’explicabilité (Grad‑CAM amélioré, variantes hiérarchiques) pour fournir des explications plus fines et standardisées aux radiologues.

Du point de vue du déploiement, une approche réaliste serait de mettre en place, dans un premier temps, un projet pilote au sein d’un service de radiologie d’Abidjan, où le modèle serait utilisé en double lecture : l’IA ne remplace pas le radiologue, mais signale les cas à haut risque (priorité à la classe *début*), avec un suivi prospectif des performances réelles. À terme, ce type d’outil pourrait contribuer à réduire les délais de dépistage, à prioriser les cas urgents et à soulager partiellement la charge des spécialistes, à condition d’être continuellement ré‑entraîné et évalué sur des données locales.Dans cette perspective, un plan de déploiement progressif spécifique au CHU d’Abidjan peut être envisagé, structuré en plusieurs phases successives, comme détaillé dans la section suivante."""

new_perspectives = """## 5.3 Perspectives : Segmentation U-Net, Cascade et Déploiement Clinique

Plusieurs pistes de travail se dégagent pour prolonger ce mémoire et aller vers un déploiement progressif dans un environnement clinique comme le CHU d’Abidjan. L'une des perspectives les plus prometteuses, testée de manière préliminaire à l'issue de ce travail, est **l'intégration d'un modèle de segmentation U-Net en cascade avec notre classifieur DenseNet-121**.

### 5.3.1 Architecture en cascade (U-Net + DenseNet) : Expérimentation Pratique

Actuellement, le modèle DenseNet-121 traite l'échographie entière. Cependant, en contexte clinique, les radiologues fondent leur diagnostic sur des critères BI-RADS précis liés à la forme de la lésion (contours spiculés, orientation, etc.). Afin d'isoler la lésion du bruit de fond (tissus sains, ombres acoustiques), nous avons développé et testé un pipeline en deux étapes :
1. **Étape de Segmentation (U-Net) :** Un réseau U-Net génère un masque binaire détourant la tumeur au pixel près.
2. **Étape de Classification (DenseNet-121) :** Le masque est utilisé pour extraire la zone d'intérêt (soit par recadrage, soit par masquage du fond) avant de la transmettre au DenseNet pour la classification finale (normal, début, grave).

**Résultats de l'expérimentation :**
Un test pratique rapide a été implémenté en masquant le fond de l'image (mise en noir de tout ce qui n'est pas la tumeur). Les résultats immédiats sur DenseNet (entraîné originellement sur des images entières) ont montré une forte chute de l'accuracy (environ 15%). Ce comportement est scientifiquement cohérent et riche en enseignements :
* **Biais de contexte global :** Il démontre que le DenseNet actuel s'appuie fortement sur la texture globale du parenchyme mammaire (le fond) pour prédire la classe, et pas uniquement sur la tumeur elle-même.
* **Nécessité d'un ré-entraînement conjoint :** Pour qu'une architecture en cascade soit performante, il est impératif de **ré-entraîner entièrement le modèle DenseNet-121 sur les images détourées (masquées par U-Net)**. Ainsi, le classifieur apprendra à n'extraire que les caractéristiques intrinsèques de la tumeur (bordures, spiculation interne) sans dépendre du contexte global.

### 5.3.2 L'apport de U-Net face aux spécificités ivoiriennes (Âge et Densité)

L'utilisation d'un U-Net ouvre également une voie majeure pour adresser le défi des tissus mammaires denses, très fréquents chez les patientes ivoiriennes jeunes. Un modèle de segmentation permet de :
* **Calculer automatiquement les caractéristiques morphologiques :** Taille exacte en cm², orientation (parallèle ou non-parallèle), et régularité des marges. Ces paramètres extraits mathématiquement offrent une explicabilité totale au médecin.
* **Isoler l'effet de l'âge :** En séparant la lésion du tissu fibro-glandulaire dense environnant, le système réduit considérablement les faux positifs chez les patientes jeunes.
* **Suivi longitudinal :** U-Net permettrait de mesurer objectivement l'évolution du volume tumoral d'une patiente sous chimiothérapie néo-adjuvante mois après mois.

### 5.3.3 Plan de déploiement progressif au CHU d’Abidjan

Sur le plan des données, la priorité est la constitution d’une base de données locale d’échographies mammaires annotées (idéalement avec des masques de segmentation validés par des radiologues), couvrant la diversité des appareils et des profils anatomiques ivoiriens. 

Du point de vue du déploiement, une approche réaliste s'organiserait en double lecture : l’IA ne remplace pas le radiologue, mais agit comme un "filet de sécurité". Le système U-Net détoure la lésion, extrait les mesures, et DenseNet propose un score de risque. Les cas signalés "début" ou "grave" par le modèle sont priorisés dans la file de relecture du médecin, réduisant ainsi les délais de prise en charge pour les patientes à haut risque."""

content = content.replace(old_perspectives, new_perspectives)

with open("Memoir_Master_classification_cancer_breast.md", "w") as f:
    f.write(content)

print("Perspectives updated successfully.")
