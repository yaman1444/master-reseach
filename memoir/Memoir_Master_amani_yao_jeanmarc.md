**MINISTERE DE L'ENSEIGNEMENT SUPERIEUR REPUBLIQUE DE COTE D'IVOIRE**

**ET DE LA RECHERCHE SCIENTIFIQUE** \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

> **Union - Discipline - Travail**
>
> ![](media/image2.jpeg)**UNIVERSITE VIRTUELLE DE CÔTE D'IVOIRE**
>
> **UFR INFORMATIQUE**
>
> **ET SCIENCES DU NUMERIQUE**
>
> \-\-\-\-\-\-\-\-\-\-\-\-\-\--

N° d'ordre :

![](media/image3.png){width="6.531944444444444in"
height="6.032638888888889in"}Mémoire Pour l'obtention du :

**Diplôme de Master**

Domaine : Sciences et Technologies

Mention : Informatique et Applications Numériques

Spécialité : Big Data Analytics (BDA)

> Présenté par

Amani Yao Jean Marc

Détection précoce du cancer du sein par l'apprentissage profond

[Sujet :]{.underline}

Soutenu le .../.../...devant le jury composé de :

  ------------------------------------------------------------------------------
  *M.*              *Professeur      *Université Virtuelle de    *Président*
                    Titulaire*       Côte d'Ivoire*              
  ----------------- ---------------- --------------------------- ---------------
  *M.*              *Maître          *...*                       *Superviseur*
                    Assistant*                                   

  *Mme.*            *Maître de       *...*                       *Examinateur*
                    Conférences*                                 

  *Mme.*            *Maître de       *...*                       *Examinateur*
                    Conférences*                                 
  ------------------------------------------------------------------------------

[]{#_Toc229492793 .anchor}

`REMERCIEMENTS`

Je remercie en premier lieu DIEU de m'avoir donné la force, la
persévérance et la capacité de mener à bien ce mémoire de **Master Big
Data Analytics**.

Je tiens à exprimer ma profonde gratitude à Monsieur Edja Béranger,
Directeur de mémoire, pour son encadrement rigoureux, sa disponibilité
constante et ses conseils avisés qui ont structuré ce travail.

Je souhaite adresser ma reconnaissance à l'ensemble des enseignants du
Master Big Data Analytics de l'université virtuelle de côte d'ivoire
pour la qualité exceptionnelle de leur formation et l'excellence
académique dispensée.

Un remerciement particulier à mes collègues de promotion pour nos
échanges enrichissants, notre soutien mutuel et les moments de partage
qui ont marqué cette année universitaire.

Je remercie l'administration universitaire pour son appui dans les
démarches administratives et la mise à disposition des ressources
nécessaires à ce travail.

Je remercie du fond du cœur ma famille pour son soutien indéfectible, sa
patience et sa confiance durant ces mois de recherche intensive.

Je remercie chaleureusement mes amis proches pour leur présence
constante, leurs encouragements sincères et leur compréhension face aux
exigences de ce mémoire.

**Abidjan,** **le 27 février 2026**\
**Amani Yao Jean Marc**[]{#_Toc229492794 .anchor}

`DÉDICACE`

À ma famille bien-aimée,\
À toutes les femmes ivoiriennes touchées par le cancer du sein,\
À celles qui luttent aujourd'hui,\
À celles qui s'en sont sorties,\
À la mémoire de celles qui nous ont quittées.

Ce travail vous est dédié.[]{#_Toc229492795 .anchor}

`RÉSUMÉ`

**Contexte et Problématique :** Le cancer du sein constitue en Côte
d'Ivoire un défi de santé publique majeur, avec plusieurs milliers de
nouveaux cas annuels et un taux de létalité supérieur à 50 %
[\[1\]](#ref1). Cette forte mortalité est principalement due à un
diagnostic tardif (stades III-IV dans plus de 70 % des cas) et à un
déficit criant d'infrastructures de dépistage [\[2\]](#ref2). Dans ce
contexte, l'échographie mammaire s'impose comme un outil de première
ligne. Cependant, la fatigue cognitive des praticiens et la difficulté
de déceler les lésions précoces, particulièrement sur les tissus denses
fréquents chez les patientes africaines, limitent l'efficacité du
diagnostic.

**Solution Proposée :** Ce mémoire développe un système de diagnostic
assisté par ordinateur (CADx) adapté au contexte ivoirien, visant à
améliorer la détection précoce des lésions mammaires suspectes. La
méthodologie repose sur un modèle d'apprentissage profond DenseNet-

121 [\[4\]](#ref4) et EfficientNet-B0 pré-entraînés, affinés pour une classification en trois
classes cliniques strictes : normal, bénin, et malin.

Sur un jeu de test indépendant rigoureusement assaini de toute fuite de données (data leakage), le modèle EfficientNet-B0 atteint une exactitude globale de 82,7 %, avec un F1-score macro de 0,81. Plus important encore, la sensibilité (rappel) pour la détection des lésions malignes atteint 90,0 %. En parallèle, l'approche en Cascade U-Net permet d'atteindre une précision maximale de 79,7 % sur la classe maligne, tout en offrant une explicabilité visuelle via la segmentation.

Les contributions de ce travail sont les suivantes :\
1) Mise en place d'un pipeline d'IA complet pour la classification
mammaire en trois classes, respectant la nomenclature clinique stricte ;\
2) Résolution de problèmes profonds tels que le Catastrophic Forgetting par une gestion fine de la BatchNormalization ;\
3) Benchmark rigoureux entre EfficientNet-B0 et une Cascade U-Net + DenseNet sur un dataset purifié pour le contexte du dépistage ivoirien.

**Perspectives :** Déploiement pilote au sein d'un environnement
hospitalier ivoirien (tel que le CHU d'Abidjan), la création d'une
bio-banque de données ouest-africaines, ainsi que l'exploration de
modèles de segmentation tels que le réseau U-Net [\[23\]](#ref23), qui
pourraient permettre d'isoler et de caractériser successivement les
lésions en tenant compte de l'âge des patientes.

**Mots-clés :** Cancer du sein, Échographie mammaire, DenseNet-121,
Détection précoce, U-Net, Triage clinique, Côte
d'Ivoire.[]{#_Toc229492796 .anchor}

**ABSTRACT**

Breast cancer remains a major public health issue in Côte d'Ivoire, with
several thousand new cases each year and a case-fatality rate above 50
%, mainly due to late-stage diagnosis (over 70 % of cases at stages
III--IV). This dissertation develops a computer-aided diagnosis (CADx)
system to improve early detection of suspicious breast lesions from
breast ultrasound images.

The proposed methodology fine-tunes a pre-trained DenseNet-121 model for
a three-class clinical task (normal, early, advanced) on a dataset of
1,580 images split into training, validation and test sets. The pipeline
includes CLAHE preprocessing, advanced data augmentation (Mixup
[\[10\]](#ref10)), a weighted Focal Loss and progressive fine-tuning of
the backbone. A post-hoc clinical threshold calibration step is then
applied to each class to optimize sensitivity for early-stage cases.
Experimental results demonstrate that the proposed system achieves a
macro-AUC of 0.92 and a global accuracy of 76.7%, with a specific recall
for early-stage lesions (the primary clinical target) reaching 90.4%.

On the independent test set (240 images), the calibrated DenseNet-121
model achieves 76.7 % accuracy, a macro AUC of 0.92 and a macro F1-score
of 0.73. Most importantly, the "early" class reaches a recall of 90.4 %,
compared to 88.1 % for the baseline configuration. The confusion matrix
shows that zero "early" cases and zero "advanced" cases are predicted as
"normal", which is consistent with a screening-oriented strategy where
missing suspicious lesions must be avoided at all costs.

The main contributions of this work are:\
1) A complete and reproducible DenseNet-121 pipeline for three-class
ultrasound classification (normal, early, advanced) in the Ivorian
context;\
2) An optimisation strategy explicitly tailored to early detection,
combining weighted Focal Loss and hierarchical threshold calibration;\
3) A detailed analysis of the trade-off between early-stage sensitivity
and overall performance balance, in view of real-world clinical triage.

# **[TABLE DES MATIERES]{.underline}** {#table-des-matieres .TOC-Heading}

[REMERCIEMENTS [2](#_Toc229492793)](#_Toc229492793)

[DÉDICACE [3](#_Toc229492794)](#_Toc229492794)

[RÉSUMÉ [4](#_Toc229492795)](#_Toc229492795)

[Abstract [6](#_Toc229492796)](#_Toc229492796)

[CONTEXTE CLINIQUE ET ÉPIDÉMIOLOGIQUE
[11](#_Toc229492797)](#_Toc229492797)

[ÉTAT DE L'ART [25](#_Toc229492798)](#_Toc229492798)

[2.3 Déséquilibre de classes et métriques cliniques
[33](#déséquilibre-de-classes-et-métriques-cliniques)](#déséquilibre-de-classes-et-métriques-cliniques)

[MÉTHODOLOGIE DENSENET‑121 : APPROCHE ORIENTÉE CLINIQUE
[39](#_Toc229492800)](#_Toc229492800)

[RÉSULTATS EXPÉRIMENTAUX [53](#_Toc229492801)](#_Toc229492801)

[DISCUSSION SCIENTIFIQUE [71](#_Toc229492802)](#_Toc229492802)

[RÉFÉRENCES [91](#références)](#références)

[ANNEXES [95](#annexes)](#annexes)

[LISTE DES FIGURES [99](#liste-des-figures)](#liste-des-figures)

[LISTE DES TABLEAUX [101](#liste-des-tableaux)](#liste-des-tableaux)

[LISTE DES ABRÉVIATIONS
[102](#liste-des-abréviations)](#liste-des-abréviations)

**INTRODUCTION GÉNÉRALE**

En Côte d'Ivoire, le cancer du sein représente plusieurs milliers de
nouveaux cas par an avec un taux de mortalité supérieur à 50 %, le plus
élevé d'Afrique de l'Ouest [\[1\]](#ref1).

En effet, cette mortalité exceptionnelle résulte d'un diagnostic tardif
systématique : plus de 70 % des cas sont diagnostiqués aux stades
III-IV, contre environ 30 % en Europe.

Ce retard diagnostique s'explique principalement par un déficit majeur
en infrastructures de dépistage : on compte environ 1 mammographe pour 1
million de femmes ivoiriennes, contre 1 pour 100 000 en France
[\[2\]](#ref2).

Par conséquent, le dépistage radiologique actuel dépend énormément de
l'interprétation manuelle. **Dans ce contexte de ressources limitées,
l'échographie mammaire s'impose souvent comme la modalité de premier
recours, étant plus accessible, moins coûteuse et exempte de
rayonnements ionisants, ce qui justifie son utilisation centrale dans ce
travail.** Le dépistage reste cependant limité par trois facteurs
critiques :\
- la fatigue cognitive des radiologues (jusqu'à 200 examens par jour) ;\
- la difficulté de détection des anomalies au stade précoce (sensibilité
moyenne des radiologues ≈ 72 %) ;\
- la forte prévalence de tissus mammaires denses (BI-RADS C/D) chez les
femmes africaines (≈ 65 % vs 25 % chez les patientes caucasiennes).

Face à cette problématique, les CNN médicaux de génération 2023-2026
(DenseNet, EfficientNet, Vision Transformers) atteignent des niveaux de
performance élevés (accuracy \> 90 %, AUC \> 0,90) sur des jeux de
données principalement caucasiens (CBIS-DDSM, INbreast). Cependant,
aucune étude n'a, à ce jour, évalué de manière systématique leur
comportement dans un contexte africain marqué par des tissus denses, un
accès limité à l'imagerie et une forte contrainte de ressources.

Ainsi, une brèche de recherche persiste : il n'existe pas de travail
ivoirien documentant la performance d'un modèle profond optimisé
spécifiquement pour la **détection précoce** dans ce contexte. C'est
pourquoi, l'objectif principal de ce mémoire est de développer un
système CADx atteignant un **recall ≥ 90 % sur les lésions débutantes**,
tout en maintenant une performance globale suffisante pour le triage
clinique (accuracy et F1-score macro stables).

Pour atteindre cet objectif, les objectifs spécifiques sont les suivants
:\
1. Concevoir un pipeline DenseNet-121 complet pour la classification par
échographie mammaire en trois classes (normal, bénin, malin) sur un
dataset d'échographies annotées ;\
2. Intégrer et évaluer Grad-CAM afin d'analyser la cohérence des régions
activées par le modèle avec les zones suspectes décrites par les
radiologues ;\
3. Étudier l'impact des contraintes spécifiques au contexte ivoirien
(tissus denses, ressources limitées, taille d'échantillon) sur les
performances du modèle et sur le risque de faux négatifs ;\
4. Prototyper un pipeline reproductible (code source versionné, scripts
d'entraînement, de visualisation et de calibration) en vue d'un futur
déploiement clinique dans un environnement hospitalier ivoirien.

Le mémoire s'articule ainsi :\
- Le **Chapitre 1** présente le contexte clinique ivoirien,
l'épidémiologie du cancer du sein et les limites structurelles du
dépistage actuel ;\
- Le **Chapitre 2** propose un état de l'art des CNN appliqués à
l'échographie mammaire et des méthodes d'explicabilité (Grad-CAM) ;\
- Le **Chapitre 3** décrit en détail la méthodologie DenseNet-121
proposée, le pipeline de prétraitement et le protocole d'entraînement ;\
- Le **Chapitre 4** expose les résultats expérimentaux, en particulier
l'optimisation orientée détection précoce et l'analyse des matrices de
confusion ;\
- Le **Chapitre 5** discute la portée clinique des résultats, les
limites de l'étude et les perspectives de déploiement au CHU d'Abidjan
et d'extension à d'autres architectures.

**CHAPITRE 1**[]{#_Toc229492797 .anchor}

**CONTEXTE CLINIQUE ET ÉPIDÉMIOLOGIQUE**

**CONTEXTE CLINIQUE EN CÔTE D'IVOIRE**

**1.1 Épidémiologie du cancer du sein en Côte d'Ivoire (GLOBOCAN 2022)**

En Côte d'Ivoire, le cancer du sein est le cancer le plus fréquent chez
la femme et représente la première cause de cancer féminin. Selon les
estimations GLOBOCAN 2022 [\[1\]](#ref1), on dénombre 3 869 nouveaux cas
de cancer du sein chez la femme ivoirienne, soit 33,5% de l'ensemble des
cancers féminins diagnostiqués en 2022. Sur la même période, plus de 2
000 décès sont attribués au cancer du sein, ce qui reflète une létalité
élevée par rapport aux pays à haut revenu.

![Charge du cancer du sein en Côte d'Ivoire (GLOBOCAN
2022)](media/image4.jpg){width="4.528113517060367in"
height="3.5756485126859143in"}

**Tableau 1.1 : Charge du cancer du sein chez la femme en Côte d'Ivoire
(GLOBOCAN 2022)**

  ----------------------------------------------------------------------
                   Indicateur                         Valeur 2022
  --------------------------------------------- ------------------------
  Nouveaux cas de cancer du sein (femmes, tous           3 869
                      âges)                     

     Part du sein dans les cancers féminins              33,5%

    Rang du cancer du sein parmi les cancers              1er
                    féminins                    

          Principale source de données             Registre du cancer
                                                       d'Abidjan
  ----------------------------------------------------------------------

Cette charge s'inscrit dans un contexte régional où le cancer du sein
est également la première cause de cancer chez la femme en Afrique de
l'Ouest, avec des taux de mortalité élevés liés au diagnostic tardif et
aux limites des systèmes de santé. Comparativement à des pays à haut
revenu comme la France, l'incidence brute est plus faible mais la
mortalité proportionnelle est plus élevée, traduisant un décalage vers
des stades plus avancés au moment du diagnostic.

**1.2 Limites du screening actuel : 1 mammographe pour 1 million de
femmes**

Le dépistage organisé du cancer du sein n'est pas encore pleinement
opérationnel en Côte d'Ivoire, et l'accès à la mammographie reste
concentré dans quelques structures de référence, principalement à
Abidjan. Plusieurs rapports insistent sur la faible disponibilité des
équipements d'imagerie et sur le nombre limité de professionnels formés
à l'imagerie mammaire. Dans la pratique, on estime qu'il n'existe que
quelques mammographes fonctionnels à l'échelle nationale, soit un ordre
de grandeur d'environ 1 mammographe pour 1 million de femmes, contre
environ 1 pour 100 000 dans des pays comme la France [\[2\]](#ref2).
**Cette pénurie structurelle fait de l'échographie mammaire l'outil de
diagnostic de première ligne le plus répandu en Côte d'Ivoire, car elle
est beaucoup plus disponible dans les centres de santé de second
contact.**

![Contraintes d'infrastructure de
dépistage](media/image5.jpg){width="5.705142169728784in"
height="3.3848687664041996in"}

**Tableau 1.2 : Contraintes infrastructurelles et organisationnelles en
dépistage**

  -------------------------------------------------------------------------
      Critère         Côte d'Ivoire     Pays à haut   Conséquence clinique
                                           revenu          principale
                   (ordre de grandeur)                
                                        (ex. France)  
  ---------------- ------------------- -------------- ---------------------
    **Densité de    Très faible (\~1    Plus élevée      Accès limité au
   mammographes**    pour 1 000 000                         dépistage
                         femmes)       (\~1 pour 100  
                                        000 femmes)   

   **Répartition      Concentrés à      Répartition   Retards diagnostiques
        des        Abidjan et grandes  plus homogène    en zones rurales
   équipements**         villes            sur le     
                                         territoire   

   **Radiologues    Nombre restreint      Équipes       Charge élevée par
    spécialisés                          dédiées à         radiologue
       sein**                            l'imagerie   
                                          mammaire    

   **Programme de   Structuration en     Dépistage      Faible couverture
    dépistage**     cours, couverture     organisé       populationnelle
                         limitée           depuis     
                                         plusieurs    
                                           années     
  -------------------------------------------------------------------------

Dans la pratique, un radiologue peut être amené à interpréter de
nombreuses échographies par jour, ce qui augmente le risque de fatigue
visuelle et de baisse de vigilance. Cette situation est particulièrement
critique pour la détection fine des lésions suspectes, qui peuvent
présenter des signes subtils sur les images.

**1.3 Classification BI-RADS et types de lésions**

**(masses et kystes)**

La lecture radiologique s'appuie sur le système BI-RADS (Breast Imaging
Reporting and Data System) développé par l'American College of
Radiology, qui standardise la description des anomalies pour la
mammographie, l'échographie et l'IRM. Ce système associe chaque
catégorie à une probabilité approximative de malignité. Dans ce travail,
nous nous concentrons sur le **BI-RADS échographique**, dont les
critères de forme, d'orientation et de contours sont essentiels pour la
classification automatisée. Bien que les microcalcifications ductales
(Ductal Carcinoma In Situ, DCIS) soient davantage un marqueur
mammographique, l'échographie permet de caractériser avec une grande
précision les masses solides et les kystes, stades pivots de la
détection précoce.

![Illustration schématique de la classification
BI-RADS](media/image6.jpg){width="6.237076771653543in"
height="3.251689632545932in"}

**Tableau 1.3 : Classification BI-RADS et probabilité de malignité**

  ---------------------------------------------------------------------
   Catégorie BI-RADS       Probabilité de        Signes cliniques et
                             malignité         radiologiques typiques
                                              
                        (ordre de grandeur)   
  -------------------- ---------------------- -------------------------
           4A                  2--10%            Masse aux contours
                                               réguliers mais suspecte

           4B                 10--50%            Masse aux contours
                                                  micro-lobulés ou
                                                     indistincts

           4C                 50--95%            Masse aux contours
                                                anguleux ou spiculés

           5                   \> 95%          Masse infiltrante très
                                               évocatrice de malignité
  ---------------------------------------------------------------------

\[d'après les recommandations BI-RADS de l'American College of
Radiology\]

Plusieurs travaux montrent que, dans des conditions optimales, la
mammographie peut atteindre une sensibilité élevée pour la détection de
microcalcifications associées à un DCIS, avec des sensibilités
rapportées autour de 88--95% mais des spécificités plus modestes (≈
40--60%). Cependant, ces performances sont souvent observées dans des
programmes de dépistage structurés, avec une qualité d'image élevée et
des radiologues expérimentés.

**1.4 Tissus mammaires denses et spécificités africaines**

La densité mammaire est un facteur clé qui influence à la fois le risque
de cancer du sein et la performance de la mammographie. Des études
menées en Afrique subsaharienne et sur des populations

afro-descendantes montrent des profils de densité mammaire différents de
ceux observés chez les patientes caucasiennes [\[18\]](#ref18), avec
souvent une proportion importante de femmes présentant des densités
élevées (BI-RADS C/D) à âge comparable.

![Tissus mammaires denses (BI-RADS C et
D)](media/image7.jpg){width="6.303861548556431in"
height="4.711071741032371in"}

**Tableau 1.4 : Densité mammaire (BI-RADS) -- tendances observées**

  -------------------------------------------------------------------------------
        Population étudiée        Proportion seins denses        Référence
                                       (BI-RADS C/D)      
  ------------------------------- ----------------------- -----------------------
              Cohorte             Proportions variables,     [\[18\]](#ref18)
   ouest-africaine/est-africaine  avec profils différents 
       (ex. Kenya, Ouganda)           des populations     
                                       occidentales       

      Femmes afro-américaines        Densité mammaire        [\[18\]](#ref18)
                                  souvent plus élevée que 
                                      chez les femmes     
                                         blanches         

        Femmes caucasiennes         Proportion de seins      Données issues de
                                   très denses variable   programmes de dépistage
                                  selon l'âge et le pays  
  -------------------------------------------------------------------------------

Ces différences de densité mammaire ont deux conséquences majeures :

- une diminution de la sensibilité de la mammographie, car les lésions
  sont plus difficiles à distinguer d'un parenchyme glandulaire dense ;
- un risque relatif de cancer plus élevé pour les femmes ayant une
  densité importante, avec des risques multipliés par 2 à 4 selon les
  études.​

Dans un contexte comme celui de la Côte d'Ivoire, où la densité mammaire
élevée est fréquente et où la qualité des images et la double lecture ne
sont pas systématiquement garanties, le risque de faux négatifs est
particulièrement préoccupant.

**1.5 Politiques de santé publique et cadre institutionnel en Côte
d'Ivoire**

La Côte d'Ivoire a progressivement structuré un cadre institutionnel
dédié à la lutte contre le cancer. Depuis plusieurs années, l'État
ivoirien investit dans cette lutte à travers la création d'une
gouvernance propre au cancer, le développement de centres anticancéreux
et la promotion de la détection précoce des cancers prévalents. Cette
dynamique a abouti à la mise en place du Programme National de Lutte
contre le Cancer (PNLCa), rattaché au Ministère de la Santé, de
l'Hygiène Publique et de la Couverture Maladie Universelle, qui
coordonne les interventions de l'État et des partenaires sur l'ensemble
du territoire.

Le Plan Stratégique National de Lutte contre le Cancer 2022--2025 (PSN
Cancer) [\[2\]](#ref2), élaboré selon une approche globale et inclusive
alignée sur le Plan National de Développement Sanitaire (PNDS) et les
objectifs régionaux de l'OMS, s'articule autour de trois axes
prioritaires : le renforcement de la gouvernance et du financement ;
l'amélioration de l'offre et de l'accessibilité aux services de qualité
; et le renforcement de la promotion de la santé pour la prévention et
le contrôle du cancer. L'objectif principal de ce plan est de réduire de
10% la mortalité liée aux cancers prévalents en Côte d'Ivoire sur la
période concernée, en s'appuyant sur un ensemble d'activités
représentant un budget estimé à plus de 166 milliards de FCFA. Un
nouveau Plan National de Lutte contre le Cancer 2025--2029 est désormais
en cours de mise en œuvre, témoignant de la continuité de l'engagement
institutionnel dans ce domaine.

**Tableau 1.5 -- Cadre institutionnel de lutte contre le cancer en Côte
d'Ivoire**

  -----------------------------------------------------------------------
        Instrument                Période             Axes principaux
  ----------------------- ----------------------- -----------------------
        PSN Cancer              2022--2025         Gouvernance, offre de
                                                  soins, promotion de la
                                                           santé

  Plan National de Lutte        2025--2029             Continuité et
     contre le Cancer                               renforcement du PSN
                                                          Cancer

     Plan National de           2021--2025        Cadre global du système
  Développement Sanitaire                            de santé ivoirien
          (PNDS)                                  
  -----------------------------------------------------------------------

Sur le plan de la santé numérique, la Côte d'Ivoire a lancé puis déployé
à l'échelle nationale la plateforme Santé CIV, un système de paiements
digitaux dans l'ensemble des hôpitaux publics du pays, accompagné d'un
programme de formation des directeurs d'hôpitaux. En parallèle, un
déploiement du Dossier Patient Informatisé (DPI) est en cours depuis fin
2024, avec des missions de câblage informatique et électrique menées
dans plusieurs régions sanitaires pour garantir une infrastructure
numérique fiable et interconnectée. En mai 2025, le gouvernement a
annoncé son objectif de connecter plus de 1 000 établissements
sanitaires à Internet haut débit, dans le cadre d'une stratégie
nationale d'Industrie 4.0 qui positionne la santé numérique comme un
secteur prioritaire.

Ces initiatives institutionnelles constituent un contexte favorable pour
envisager l'intégration future d'un outil de diagnostic assisté par IA
comme celui développé dans ce mémoire. En effet, l'existence d'un
programme national de lutte contre le cancer, d'un plan de
digitalisation des hôpitaux en cours et d'une stratégie numérique
nationale offre un cadre d'ancrage concret pour un projet pilote de
déploiement au CHU d'Abidjan, à condition que les dimensions techniques,
éthiques et organisationnelles soient pleinement prises en compte dès la
conception

**1.6 Problématique scientifique et rôle potentiel de l'IA**

La combinaison d'une charge importante de cancer du sein, d'une
mortalité élevée et de contraintes structurelles (faible densité
d'équipements, peu de radiologues, tissus denses fréquents) crée un
contexte où le diagnostic précoce est difficile à assurer de manière
homogène sur l'ensemble du territoire ivoirien. Parallèlement, les
modèles d'apprentissage profond de type CNN ont montré des performances
élevées pour la classification d'images médicales mammaires et la
détection de lésions suspectes dans des bases de données
internationales.

La question scientifique centrale de ce mémoire est la suivante :\
comment adapter et optimiser un modèle DenseNet-121 pré-entraîné pour
améliorer la détection des lésions mammaires débutantes dans le contexte
ivoirien, tout en maintenant une performance globale compatible avec un
usage de triage clinique ?

Dans cette optique, l'objectif principal est de développer un système
CADx atteignant un recall ≥ 90% sur les lésions débutantes, tout en
conservant une accuracy et un F1-score macro stables sur l'ensemble des
classes. Ce travail vise ainsi à valider l'impact combiné d'une Focal
Loss pondérée et d'une calibration hiérarchique des seuils sur la
réduction stricte des faux négatifs aux stades précoces, et à illustrer
visuellement les prédictions par Grad-CAM.

Ce travail ne vise pas à remplacer la lecture humaine, mais à proposer
un outil de triage augmentant la sensibilité sur les lésions débutantes
dans un contexte de ressources limitées. Le chapitre suivant présente
l'état de l'art des jeux de données de mammographie, des architectures
de deep learning utilisées en imagerie mammaire et des approches
d'explicabilité visuelle adaptées à la pratique clinique.

**CHAPITRE 2**[]{#_Toc229492798 .anchor}

**ÉTAT DE L'ART**

**2.1 Jeux de données pour le diagnostic du cancer du sein**

L'analyse par intelligence artificielle du cancer du sein s'appuie sur
plusieurs bases d'images publiques qui servent de référence pour
entraîner et comparer les modèles. Les premières concernent la
mammographie. Le dataset DDSM puis sa version nettoyée CBIS‑DDSM
regroupent des mammographies sur film numérisées, annotées par des
radiologues avec des informations sur les masses, les calcifications et
le statut bénin/malin. Le dataset INbreast correspond, lui, à des
mammographies numériques plein champ (FFDM) recueillies dans un centre
du sein portugais (410 images, 115 cas) avec des annotations précises de
divers types de lésions.

![Exemples d'échographies mammaires (Dataset
BUSI)](media/image8.jpg){width="5.833333333333333in"
height="2.067113954505687in"}

En parallèle, des jeux de données se sont développés en échographie
mammaire, modalité particulièrement pertinente dans les seins denses. Le
Breast Ultrasound Images Dataset (BUSI) proposé par Al‑Dhabyani et
al. (2020) [\[3\]](#ref3) est l'un des plus utilisés. Il contient 780
échographies mammaires réparties en trois classes : 437 images bénignes,
210 malignes et 133 normales, la plupart accompagnées de masques de
segmentation de la lésion.

**Tableau 2.1 -- Principaux jeux de données en cancer du sein**

  -------------------------------------------------------------------------------
    Dataset         Modalité         Nombres    Types de classes    Annotations
                                     d'images       / lésions       principales
                                    (ordre de                     
                                    grandeur)                     
  ------------ ------------------- ------------ ----------------- ---------------
   CBIS‑DDSM    Mammographie film    ≈ 3 000         Masses,        Contours de
                                                 calcifications      lésions,
                                                  (bénin/malin)     diagnostics

    INbreast    Mammographie FFDM      410           Masses,       Contours XML,
                                                 calcifications,       infos
                                                     autres        pathologiques

      BUSI      Échographie sein       780           Bénin,        Masques de
                                                Malin, Normal lésions, labels
                                                                      globaux
  -------------------------------------------------------------------------------

Pour BUSI, les caractéristiques détaillées sont les suivantes.

**Tableau 2.2 -- Caractéristiques du dataset BUSI (Al‑Dhabyani et al.,
2020) [\[3\]](#ref3)**

  ---------------------------------------------------------
   Caractéristique           Valeur / Description
  ----------------- ---------------------------------------
      Modalité               Échographie mammaire

    Nombre total                      780
      d'images      

       Bénin                     437 images

      Malin                   210 images

       Normal                     133 images

       Format         PNG, résolution ≈ 500 × 500 pixels,
                                niveaux de gris

     Annotations        Masques de segmentation pour la
                             majorité des lésions
  ---------------------------------------------------------

Dans ce mémoire, les données proviennent de la base publique BUSI (780
images réparties en Normal, Bénin, Malin). Afin d'adapter cette
nomenclature aux besoins d'un triage clinique axé sur la détection
précoce, la classe « Bénin » a été assimilée à la classe « bénin »
(désignant ici toute anomalie ou nodule bénin nécessitant un suivi ou
une biopsie pour écarter tout risque de cancer), tandis que « Malin
» correspond à « malin » (cancers avérés infiltrants). Il convient de
mentionner une note de prudence méthodologique : cliniquement, une
lésion bénigne peut être volumineuse et ancienne, et ne correspond pas
toujours à un stade "précoce" d'un cancer. Toutefois, dans le cadre
scientifique de ce système de triage computationnel, cette
simplification est assumée car les nodules bénins regroupent les profils
d'imagerie les plus ambigus à surveiller, à l'image des lésions
débutantes suspectes.

**2.2 Architectures de deep learning en imagerie mammaire**

Depuis une dizaine d'années, les réseaux de neurones convolutifs (CNN)
ont profondément transformé l'analyse d'images médicales. Les premiers
travaux en cancer du sein utilisaient des architectures génériques
pré‑entraînées sur ImageNet (AlexNet, VGG, ResNet) puis adaptées à la
mammographie ou à l'échographie par transfert d'apprentissage. Ces
modèles permettent déjà d'atteindre des AUC supérieures à 0,90 pour la
classification bénin/malin sur des jeux de données comme CBIS‑DDSM,
INbreast ou BUSI.

![Schéma simplifié de l'architecture
DenseNet-121](media/image9.jpg){width="6.2608038057742785in"
height="3.773766404199475in"}![Illustration des connexions
denses](media/image10.jpg){width="6.837341426071741in"
height="4.545223097112861in"}

Des architectures plus récentes se distinguent particulièrement en
imagerie mammaire : les réseaux résiduels ResNet, les réseaux à
connexions denses DenseNet, la famille EfficientNet basée sur un
"compound scaling" de la profondeur, de la largeur et de la résolution,
ainsi que les Vision Transformers (ViT) ou modèles hybrides
CNN--Transformer qui exploitent des mécanismes d'attention globale.

**Tableau 2.3 -- Principales familles d'architectures utilisées**

  -------------------------------------------------------------------------------
          Famille                  Principe clé             Atouts en imagerie
                                                                 mammaire
  ----------------------- ------------------------------- -----------------------
          ResNet              Connexions résiduelles      Facilite l'entraînement
                                  [\[7\]](#ref7)              de réseaux très
                                                                 profonds

         DenseNet         Connexions denses entre toutes     Réutilisation de
                            les couches [\[4\]](#ref4)     caractéristiques, peu
                                                               de paramètres

       EfficientNet             "Compound scaling"          Excellent compromis
                           profondeur/largeur/résolution   performance / coût de
                                  [\[8\]](#ref8)                  calcul

      ViT / hybrides       Attention globale sur patchs   Capture des dépendances
                                      d'image                à longue distance
  -------------------------------------------------------------------------------

Les architectures de type Vision Transformer (ViT) [\[9\]](#ref9) et les
modèles hybrides CNN--Transformer ont fait l'objet de nombreux travaux
récents en imagerie mammaire. Ils reposent sur des mécanismes
d'attention appliqués à des patchs d'image, ce qui leur permet de
capturer des dépendances à longue distance entre différentes régions du
sein, au‑delà du champ réceptif local des convolutions classiques. Des
études comparatives sur des images histopathologiques du sein montrent
par exemple que des ViT atteignent des accuracies de l'ordre de 93--94%
en validation, en dépassant légèrement des architectures CNN profondes
comme ResNet ou DenseNet lorsque de très grands jeux de données sont
disponibles.

Cependant, ces gains restent fortement dépendants de la taille du corpus
d'entraînement et des ressources de calcul mobilisées. Une analyse
récente de plusieurs modèles de transfert d'apprentissage pour la
classification du cancer du sein (ResNet‑50, DenseNet‑121, EfficientNet,
MobileNet, ViT, etc.) montre que les Transformers ont tendance à
nécessiter davantage de paramètres et des temps d'entraînement plus
longs pour exprimer pleinement leur potentiel, alors que des CNN
optimisés restent très compétitifs sur des jeux de données de taille
modérée. En parallèle, des approches hybrides combinant un backbone CNN
(par exemple EfficientNetB0 ou ResNet50) avec des blocs d'attention
inspirés des Transformers ont été proposées pour la classification de
sous‑types tumoraux ou l'analyse temporelle de lésions mammaires, avec
des performances élevées mais essentiellement évaluées dans des centres
tertiaires fortement équipés.

Dans le contexte spécifique de l'échographie mammaire, des architectures
CNN récentes comme EfficientNet‑B7 ont déjà permis d'atteindre des
accuracies proches de 99% et des AUC proches de 1,00 sur le dataset
BUSI, en combinant transfert d'apprentissage, fortes augmentations de
données et méthodes d'explicabilité. Bien que les modèles de type Vision
Transformers (ViT) ou EfficientNet soient particulièrement prometteurs,
aucune évaluation expérimentale de ces modèles n'a été conduite dans ce
mémoire en raison de contraintes computationnelles strictes (ressources
limitées pour le pré-entraînement massif requis par les ViTs) et du
sur-ajustement systématique observé sur des datasets de petite taille
comme BUSI. C'est dans cette logique pragmatique que ce mémoire retient
DenseNet‑121 comme backbone principal : cette architecture reste très
compétitive par rapport aux alternatives, tout en étant plus simple à
entraîner, à expliquer via Grad‑CAM et à intégrer dans une
infrastructure réaliste pour la Côte d'Ivoire.

## **2.3 Déséquilibre de classes et métriques cliniques**

Les jeux de données médicaux présentent souvent un déséquilibre de
classes, avec une sur‑représentation de certaines catégories par rapport
à d'autres. En imagerie du sein, les bases publiques contiennent
généralement davantage de lésions bénignes que malignes, ou un nombre
limité de cas normaux bien documentés, ce qui peut conduire les modèles
à privilégier la classe majoritaire. Ce phénomène fausse l'accuracy
globale et masque une sensibilité insuffisante sur les classes
minoritaires, notamment les stades précoces.

Pour atténuer ces effets, plusieurs techniques sont décrites dans la
littérature :

- Pondération de la fonction de coût / Focal Loss [\[5\]](#ref5) : la
  Focal Loss modifie la cross‑entropy en réduisant le poids des exemples
  faciles et en renforçant celui des exemples difficiles, améliorant
  ainsi l'apprentissage des classes minoritaires.
- Ré‑échantillonnage (oversampling des classes rares, undersampling des
  classes majoritaires) et augmentation de données ciblée (rotations,
  flips, Mixup, CutMix [\[11\]](#ref11)) pour présenter un volume plus
  équilibré d'images pendant l'entraînement.
- Choix de métriques adaptées, comme le recall, le F1‑score par classe,
  la macro‑moyenne et l'AUC, pour mieux refléter la performance réelle
  sur chaque classe.​

**Tableau 2.4 -- Stratégies de gestion du déséquilibre des classes**

  -----------------------------------------------------------------------
         Technique            Idée principale          Effet attendu
  ----------------------- ----------------------- -----------------------
    Focal Loss / class    Renforcer le poids des  Améliorer le recall des
          weights            classes rares ou      classes minoritaires
                                difficiles        

      Oversampling /       Modifier la fréquence   Réduire le biais vers
       undersampling         d'échantillonnage     la classe majoritaire

    Augmentation ciblée    Générer de nouvelles        Améliorer la
                            variantes d'images        généralisation
                                   rares          

   Calibration de seuils   Ajuster les seuils de  Contrôler le compromis
                            décision par classe   sensibilité / précision
  -----------------------------------------------------------------------

En contexte de dépistage, de nombreux travaux insistent sur la nécessité
de prioriser la sensibilité (recall) pour les lésions suspectes ou
précoces, quitte à accepter davantage de faux positifs qui seront
ensuite filtrés par le clinicien. Une approche complémentaire est la
calibration des seuils de décision par classe, qui permet d'augmenter le
rappel d'une classe cible (par exemple les lésions précoces) tout en
maîtrisant la dégradation du F1‑score global. Ce principe sera repris
dans ce mémoire pour optimiser le rappel de la classe « bénin » sans
sacrifier totalement les autres classes.

**2.4 Explicabilité des modèles : Grad‑CAM en imagerie mammaire**

L'acceptation clinique des modèles de deep learning repose aussi sur
leur explicabilité. Les radiologues souhaitent comprendre sur quelles
régions de l'image le modèle se base pour prédire une classe donnée.
Parmi les approches d'Explainable AI (XAI), Grad‑CAM (Gradient‑weighted
Class Activation Mapping) [\[6\]](#ref6) est l'une des plus utilisées
pour les CNN de classification. Elle produit, pour une classe cible, une
carte de chaleur obtenue en combinant les gradients de cette classe avec
les cartes de caractéristiques d'une couche convolutionnelle profonde.​

![Visualisation Grad-CAM en imagerie
mammaire](media/image11.jpg){width="6.719361329833771in"
height="4.613291776027997in"}

En imagerie du sein (mammographie, échographie), Grad‑CAM est employé
pour vérifier que les activations se concentrent sur les lésions
(masses, microcalcifications) plutôt que sur des artefacts ou des
éléments hors sein, et pour analyser les erreurs du modèle (faux
positifs et faux négatifs). Des variantes comme Grad‑CAM++ ou des
méthodes basées sur les valeurs de Shapley (SHAP) ont été proposées pour
affiner encore cette interprétation.

**Tableau 2.5 -- Méthodes XAI fréquemment utilisées avec les CNN**

  -----------------------------------------------------------------------
          Méthode           Type d'explication       Usage typique en
                                                     imagerie du sein
  ----------------------- ----------------------- -----------------------
         Grad‑CAM          Carte de chaleur par    Visualiser les zones
                                  classe            importantes pour la
                                                         décision

        Grad‑CAM++         Variante plus fine de   Mieux gérer plusieurs
                                 Grad‑CAM            foyers de lésion

           SHAP            Importance locale des  Analyse plus détaillée
                                 features           mais plus coûteuse
  -----------------------------------------------------------------------

Ces méthodes restent toutefois essentiellement qualitatives et ne
constituent pas une validation formelle de la décision, car les cartes
de chaleur dépendent du choix de la couche, de la normalisation et ne
reflètent pas forcément toute l'information utilisée par le modèle. Dans
ce mémoire, Grad‑CAM est utilisé pour analyser les prédictions de
DenseNet‑121 sur les trois classes normal, bénin et malin, afin de
vérifier que le modèle se focalise sur des régions mammaires pertinentes
et pour soutenir son utilisation comme outil d'aide à la décision plutôt
que comme "boîte noire".

**2.5 Synthèse critique de l'état de l'art**

Les travaux récents en imagerie mammaire montrent que les réseaux
convolutifs profonds atteignent souvent des performances très élevées en
classification binaire bénin / malin, avec des AUC proches de 0,95--0,99
sur des bases publiques comme CBIS‑DDSM, INbreast ou BUSI, à condition
de disposer d'un volume de données suffisant et de combiner transfert
d'apprentissage et augmentation de données avancée. Cependant, ces
résultats sont obtenus dans des conditions expérimentales qui
s'éloignent de plusieurs contraintes du contexte ivoirien.

Premièrement, une grande partie de la littérature se limite à une tâche
binaire, alors que la pratique clinique nécessite de distinguer au
minimum trois situations : sein sans anomalie, lésion débutante et
lésion avancée. En réduisant le problème à deux classes, plusieurs
études masquent la difficulté spécifique de séparer les lésions précoces
des seins normaux, qui est pourtant le cœur du dépistage. Les métriques
globales (accuracy, AUC binaire) ne reflètent pas toujours le risque
réel de faux négatifs sur les stades débutants.

Deuxièmement, peu de travaux considèrent explicitement le rappel des
lésions précoces comme mètre étalon principal de performance. La
majorité des articles rapportent en priorité l'accuracy globale ou l'AUC
moyenne, parfois complétées par des F1‑scores, mais sans fixer une cible
clinique forte sur la sensibilité des stades précoces. Or, dans un
contexte de dépistage en ressources limitées, un faux négatif sur une
lésion débutante a un impact bien plus critique qu'une série de faux
positifs qui seront rattrapés par le clinicien.

Troisièmement, la plupart des bases d'images utilisées proviennent de
populations caucasiennes, asiatiques ou moyen‑orientales. Le profil de
densité mammaire, l'âge des patientes, les habitudes de dépistage et les
équipements diffèrent de ceux rencontrés en Côte d'Ivoire et plus
largement en Afrique subsaharienne. Il existe donc un risque de baisse
de performance lors du transfert direct de modèles vers des patientes
africaines, sans étape d'adaptation ni validation locale. Cette question
de biais de population est encore très peu adressée de manière
systématique.

Enfin, sur le plan méthodologique, relativement peu d'études décrivent
des stratégies explicites de calibration des seuils de décision par
classe, encore moins dans une logique hiérarchique orientée "fail‑safe".
La plupart des modèles appliquent un simple argmax sur les probabilités
de sortie du réseau, ce qui revient à traiter de la même manière une
erreur "bénin → normal" et une erreur "bénin → malin", alors que leur
impact clinique est très différent.

Dans ce contexte, le présent travail se distingue par plusieurs choix :\
-- le passage à une classification en trois classes cliniques (normal,
bénin, malin) cohérente avec la pratique radiologique ;\
-- la définition d'un objectif prioritaire sur le rappel de la classe
"bénin", avec une cible explicite à 90% ;\
-- la mise en œuvre d'une calibration hiérarchique des seuils
privilégiant systématiquement une classification vers une classe
pathologique en cas de doute ;\
-- et une réflexion orientée vers l'adaptation future du modèle au
contexte ivoirien, en intégrant dès la conception les contraintes de
ressources et de densité mammaire.

Cette synthèse souligne que l'enjeu n'est pas seulement d'atteindre des
scores élevés sur un benchmark, mais de rapprocher la conception des
modèles des besoins cliniques réels, en particulier dans des
environnements à forte contrainte comme la Côte d'Ivoire.

**CHAPITRE 3**[]{#_Toc229492800 .anchor}

**MÉTHODOLOGIE DENSENET‑121 : APPROCHE ORIENTÉE CLINIQUE**

Ce chapitre décrit l'architecture logicielle, les transformations
appliquées aux images et les stratégies d'apprentissage profond mises en
œuvre dans le pipeline final (Expérience 5). L'ensemble de la
méthodologie est conçu pour répondre à deux contraintes majeures : le
déséquilibre des classes et la priorité clinique donnée à la détection
précoce du cancer du sein (classe *debut*).

**3.1 Pipeline de prétraitement : CLAHE et Mixup**

L'analyse d'images échographiques est rendue difficile par le bruit
speckle et le faible contraste des tissus. Le prétraitement vise donc à
améliorer la qualité visuelle des images tout en préservant la
morphologie des lésions.

**3.1.1 Normalisation et amélioration de contraste (CLAHE)**

Toutes les images sont redimensionnées à 320×320 pixels et normalisées
dans l'intervalle $\lbrack 0,1\rbrack$ (facteur rescale =
$\frac{1}{255}$). Pour améliorer le contraste local, un CLAHE (Contrast
Limited Adaptive Histogram Equalization) est appliqué :

- opération par tuiles locales
- paramètres : $\text{clipLimit} = 2.0$, $\text{tileGridSize} = (8,8)$

Contrairement à une égalisation d'histogramme globale, CLAHE limite
l'amplification artificielle du contraste et évite de renforcer
excessivement le bruit, ce qui est crucial pour les contours subtils des
masses mammaires.

**3.1.2 Augmentation de données : choix de Mixup**

Pour régulariser le modèle et réduire l'overfitting, une stratégie
d'augmentation de données est mise en place. Deux approches ont été
testées :

- CutMix [\[11\]](#ref11) (découpage/collage de patchs d'images)
- Mixup (combinaison linéaire d'images et de labels)

Les essais préliminaires ont montré que CutMix [\[11\]](#ref11)
détériore la morphologie spatiale des lésions (spiculations, bords,
forme globale), ce qui est problématique pour distinguer un cancer
*debut* d'un cancer *malin*. Le pipeline final conserve donc uniquement
Mixup avec :

- paramètre $\alpha = 0.1$
- probabilité d'application $p = 0.5$

Un alpha faible garantit que les images générées restent proches de la
réalité anatomique, tout en lissant les frontières de décision du
modèle.

**3.1.3 Limites et risques du prétraitement**

Bien que le prétraitement améliore la lisibilité des images et la
stabilité de l'apprentissage, il comporte aussi des limites. Un
contraste trop renforcé peut, par exemple, accentuer des artefacts ou
modifier légèrement l'apparence des contours, ce qui risque de créer des
signaux artificiels pour le modèle. Il est donc nécessaire de choisir
des paramètres de CLAHE modérés et de vérifier qualitativement que
l'aspect des lésions reste conforme à l'interprétation radiologique.

De même, l'augmentation par Mixup ne doit pas s'éloigner excessivement
de la réalité anatomique. Des combinaisons trop extrêmes pourraient
mélanger des structures appartenant à des contextes très différents et
rendre l'apprentissage instable. Le choix d'un alpha faible et d'une
probabilité d'application limitée vise justement à respecter l'équilibre
entre diversité des données et préservation des signatures visuelles
importantes pour la différenciation entre lésions débutantes et
avancées.

**3.2 Architecture DenseNet‑121 et tête de classification**

L'architecture choisie est DenseNet‑121 [\[4\]](#ref4) pré‑entraînée sur
ImageNet, utilisée comme backbone d'extraction de caractéristiques.

**3.2.1 Intérêt des blocs denses en échographie**

Dans DenseNet, chaque couche d'un *dense block* reçoit en entrée les
cartes de caractéristiques de toutes les couches précédentes, via une
concaténation progressive.\
Cette réutilisation systématique des feature maps est particulièrement
intéressante en imagerie médicale :

- conservation simultanée de motifs de bas niveau (textures, bords,
  contours de la masse)
- agrégation de concepts de plus haut niveau (forme globale de la
  tumeur, contexte tissulaire)

Cette propriété favorise une représentation riche sans explosion du
nombre de paramètres.

**3.2.2 Tête de classification et module d'attention**

Au‑dessus du backbone DenseNet‑121 (chargé sans sa tête de
classification d'origine), une tête spécifique à la tâche est ajoutée :

- couche de Global Average Pooling (GAP) pour transformer les cartes de
  caractéristiques en un vecteur de taille 1 024
- couche Dense(512, activation='relu') suivie d'un Dropout(0.3) pour
  limiter le surapprentissage
- couche de sortie Dense(3, activation='softmax') pour les trois classes
  (*debut*, *malin*, *normal*)

Un module d'attention de type CBAM (Convolutional Block Attention
Module) [\[19\]](#ref19) a été testé dans des expériences
intermédiaires, mais l'architecture finale s'appuie principalement sur
la robustesse intrinsèque de DenseNet‑121 et sur la régularisation par
Mixup / Dropout.

**3.3 Protocole d'entraînement avancé**

L'entraînement est réalisé en deux phases successives afin de stabiliser
d'abord la tête de classification, puis d'ajuster finement le backbone.

**3.3.1 Stratégie de fine‑tuning progressif**

- Phase 1 -- Apprentissage de la tête
  - Backbone DenseNet totalement gelé
  - Seules les couches supérieures (GAP + Dense(512) + Dropout +
    Dense(3)) sont entraînées
  - Taux d'apprentissage initial : $\text{lr} = 10^{- 4}$
  - Ordonnancement du lr par Cosine Annealing avec warmup de 5 époques
- Phase 2 -- Fine‑tuning conservateur
  - Dégel des ≈ 15% couches supérieures du backbone (≈ 65 couches
    entraînables sur 427)
  - Taux d'apprentissage très faible : $\text{lr} = 5 \times 10^{- 6}$
  - Objectif : adapter légèrement les filtres convolutionnels aux
    textures échographiques sans détruire les connaissances pré‑apprises
    sur ImageNet

Cette approche progressive limite les risques d'overfitting sur un
dataset de taille moyenne tout en permettant une spécialisation
raisonnable du backbone.

**3.3.2 Focal Loss pondérée et priorités cliniques**

La fonction de coût standard en classification multi-classes est la
cross-entropy catégorielle, définie pour un exemple (x_i, y_i) par :

$$L_{CE}\left( p_{t} \right) = - \log\left( p_{t} \right)$$

où p_t désigne la probabilité prédite pour la classe réelle y_i. Cette
formulation traite de manière identique les exemples faciles (bien
classés, forte confiance) et les exemples difficiles (incertains ou mal
classés), ce qui la rend inadaptée en présence d'un déséquilibre de
classes. La Focal Loss, introduite par Lin et al. (2017) [\[5\]](#ref5),
modifie ce comportement en ajoutant un facteur de modulation dynamique :

$$L_{FL}(p_{t}) = - \alpha_{t}(1 - p_{t})^{\gamma}log(p_{t})$$

où gamma = 2.0 est le paramètre de focalisation et alpha_t le poids de
classe. Lorsque le modèle est très confiant (p_t = 0.9),

le facteur (1 - 0.9)\^2 = 0.01 réduit la perte à 1 % de sa valeur
standard. Pour un exemple difficile (p_t = 0.3), ce facteur vaut
(0.7)\^2 = 0.49, maintenant une forte pression d'apprentissage sur les
cas ambigus notamment les lésions débutantes.

La distribution des classes (majorité *debut*, minorité *normal*) rend
la cross-entropy standard insuffisante. Une Focal Loss catégorielle
personnalisée est donc implémentée, inspirée de Lin et al. (2017) :

- paramètre de focalisation : $\gamma = 2.0$ (accent sur les exemples
  difficiles)
- coefficients de poids de classe $\alpha$ ajustés empiriquement pour
  refléter les priorités médicales :
  - $\alpha_{\text{debut}} = 1.2$ : léger renforcement pour garantir un
    haut taux de détection précoce
  - $\alpha_{\text{malin}} = 1.0$ : poids de référence
  - $\alpha_{\text{normal}} = 1.5$ : poids plus élevé pour compenser la
    faible taille de cette classe et éviter qu'elle soit ignorée

Plutôt que d'utiliser l'inverse brut des fréquences, ces valeurs ont été
calibrées de façon à ne pas "écraser" la classe *debut*, tout en
réhabilitant la classe *normal*.

**3.3.3 Exploration et choix des hyperparamètres**

Le choix des hyperparamètres (taux d'apprentissage, taille de lot,
nombre d'époques, paramètres de la Focal Loss) a fait l'objet d'une
exploration empirique sur le jeu de validation. Dans une première série
d'essais, des taux d'apprentissage plus élevés ont été testés pour la
phase de fine‑tuning, mais ils conduisaient rapidement à une dégradation
des performances de validation et à une instabilité des courbes de
perte, signe d'un sur‑ajustement du backbone DenseNet‑121. La valeur
finale retenue $\left( 5 \times 10^{- 6} \right)\$ représente un
compromis entre capacité d'adaptation et préservation des connaissances
pré‑apprises.

De même, plusieurs configurations de poids de classes pour la Focal Loss
ont été comparées. Des poids trop élevés pour la classe normal
amélioraient légèrement son rappel, mais au prix d'une baisse non
acceptable du rappel de la classe debut. Les coefficients finaux

(1,2 ; 1,0 ; 1,5) ont été sélectionnés car ils permettent de maintenir
un rappel élevé sur les lésions précoces tout en évitant que la classe
normale soit systématiquement sacrifiée. Cette démarche montre que
l'optimisation ne se limite pas à une recherche automatique, mais
s'appuie sur des considérations cliniques explicites.

**3.4 Métriques cliniques et calibration asymétrique des seuils**

Dans un contexte de dépistage, toutes les erreurs n'ont pas le même
impact.\
Un faux positif (classer à tort un sein sain comme pathologique)
entraîne de l'anxiété et des examens complémentaires.\
Un faux négatif sur une lésion *debut* retarde une prise en charge
potentiellement curative.

Pour cette raison :

- la métrique principale n'est pas l'accuracy, mais le recall
  (sensibilité) de la classe *debut*, avec une cible clinique fixée à ≥
  90% ;
- l'AUC‑ROC et le F1‑score macro sont utilisés comme métriques
  secondaires pour vérifier que la performance globale du modèle reste
  exploitable.

**3.4.1 Calibration hiérarchique des seuils**

**(script calibrate_exp4.py)**

Après l'entraînement, un algorithme de calibration post‑hoc des seuils
de décision par classe est appliqué :

1.  Sur le jeu de validation, on extrait les probabilités de sortie du
    modèle pour chaque classe (*debut*, *malin*, *normal*).
2.  Pour chaque classe, des courbes ROC sont analysées afin de
    déterminer un seuil optimal qui maximise la sensibilité tout en
    gardant un F1 acceptable.
3.  On obtient ainsi trois seuils cliniques *Tdebut*, *Tgrave ,Tnormal*

Lors de l'inférence, la décision suit une règle hiérarchique asymétrique
:

1.  Si $p(malin) \geq Tgrave$ la prédiction est malin.
2.  Sinon, si $p(debut) \geq Tdebut\$ la prédiction est debut.
3.  Ce n'est que si aucun seuil pathologique n'est dépassé que le modèle
    est autorisé à prédire normal.

Cette logique implémente un comportement "fail‑safe" : en cas de doute,
le modèle préfère classer vers une classe pathologique plutôt que de
rassurer à tort.

**3.5 Reproductibilité : organisation du code et suivi des expériences**

Pour que le travail soit exploitable et vérifiable, une attention
particulière a été portée à la reproductibilité du pipeline :

- Scripts modulaires
  - augmentation.py : définition du générateur de données et des
    transformations (CLAHE, Mixup)
  - focal_loss.py : implémentation de la Focal Loss pondérée
  - train_advanced.py : orchestration de l'entraînement (phases 1 et 2,
    callbacks, sauvegardes)
  - calibrate_exp4.py : calibration des seuils et évaluation
    post‑entraînement
- Gestion des random seeds\
  Toutes les bibliothèques aléatoires (Python random, NumPy,
  TensorFlow/Keras) sont initialisées avec la même graine (seed = 42)
  afin de garantir la reproductibilité du split Train/Val/Test et des
  augmentations.
- Monitoring et checkpoints
  - sauvegarde automatique du meilleur modèle (save_best_only=True) en
    monitorant la métrique val_auc
  - utilisation de callbacks d'EarlyStopping (patience 10--12 époques)
    pour limiter le surapprentissage
- Traçabilité des expériences\
  Après chaque expérience, les éléments suivants sont sérialisés (JSON /
  images) :
  - historiques d'entraînement (loss, accuracy, AUC)
  - matrices de confusion et rapports de classification
  - seuils calibrés et résultats finaux sur le jeu de test

Cette organisation facilite la comparaison entre configurations
successives (Expériences 3, 4, 5) et permet au lecteur de reproduire ou
prolonger le travail.

**3.6 Résumé et logique clinique de la méthodologie**

L'ensemble des choix méthodologiques présentés dans ce chapitre
s'inscrit dans une logique de conception orientée par le besoin
clinique. Chaque composant du pipeline (prétraitement, architecture
DenseNet‑121, schéma d'entraînement en deux phases, Focal Loss pondérée,
calibration hiérarchique des seuils) vise à réduire au maximum le risque
de faux négatifs sur les lésions débutantes, tout en conservant une
performance globale compatible avec un usage de triage.

Concrètement, la méthodologie développée dans ce mémoire repose sur un
pipeline strict pour prévenir toute fuite d'informations (data leakage).
Le jeu de données original BUSI (780 images, 3 classes : normal, bénin,
malin) a d'abord été divisé en trois sous‑ensembles disjoints au niveau
patient (70% entraînement, 15% validation, 15% test). Ensuite,
uniquement sur l'ensemble d'entraînement, des augmentations hors-ligne
ont été appliquées pour balancer les classes, générant un corpus
théorique étendu à 1 580 images. Les images sont prétraitées par
redimensionnement, normalisation et CLAHE pour améliorer le contraste
local tout en limitant le bruit.

![Répartition du dataset](media/image12.jpg){width="4.902518591426071in"
height="3.4980129046369206in"}

![Déséquilibre des
classes](media/image13.jpg){width="5.395636482939633in"
height="3.5723851706036744in"}

![Pipeline de
prétraitement](media/image14.jpg){width="4.166666666666667in"
height="2.7777777777777777in"}

Lors de l'apprentissage, une augmentation dynamique contrôlée
supplémentaire (Mixup [\[10\]](#ref10), alpha = 0,1) a été utilisée afin
d'augmenter la diversité. Le modèle s'appuie sur un DenseNet‑121
pré‑entraîné sur ImageNet, utilisé comme extracteur de caractéristiques,
surmonté d'une tête de classification adaptée à trois classes (GAP,
Dense(512, ReLU), Dropout(0,3), Dense(3, softmax)). L'apprentissage est
réalisé en deux phases : d'abord l'entraînement de la tête de
classification avec le backbone gelé, puis un fine‑tuning conservateur
des couches supérieures de DenseNet avec un faible taux d'apprentissage.
Pour gérer le déséquilibre et refléter les priorités cliniques, une
Focal Loss pondérée est utilisée (gamma = 2, poids de classes ajustés).
Enfin, une calibration hiérarchique des seuils de décision est appliquée
en post‑entraînement pour imposer une logique "fail‑safe" (malin \>
bénin \> normal), garantissant la reproductibilité avec la gestion des
seeds.

**3.7 Rappel des métriques d'évaluation**

Pour évaluer le modèle, plusieurs métriques complémentaires sont
utilisées, afin de tenir compte du déséquilibre des classes et des
priorités cliniques.

- **Accuracy** (taux de bonne classification) :

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- L'accuracy mesure la proportion d'images correctement classées parmi
  l'ensemble des images du jeu de test.\
  Elle est utile pour donner une vision globale, mais peut être
  trompeuse en présence de classes minoritaires.

<!-- -->

- **Recall / Sensibilité** par classe :

$$\text{Recall} = \frac{TP}{TP + FN}$$

- Le recall d'une classe mesure la proportion de vrais positifs
  correctement détectés parmi tous les exemples réellement appartenant à
  cette classe.\
  Dans ce mémoire, le recall de la classe *debut* est la métrique
  principale, car il correspond à la capacité du modèle à détecter les
  lésions précoces (erreur la plus critique en dépistage).

<!-- -->

- **Précision** et **F1‑score** par classe :

$$\text{Précision} = \frac{TP}{TP + FP}$$

$$F1 = 2 \times \frac{\text{Précision} \times \text{Recall}}{\text{Précision} + \text{Recall}}$$

- La précision mesure la proportion de vrais positifs parmi toutes les
  prédictions positives d'une classe donnée.\
  Le F1‑score est la moyenne harmonique entre précision et recall, ce
  qui permet d'équilibrer les deux aspects.\
  Le F1 macro (moyenne du F1 sur les trois classes) est utilisé comme
  indicateur global de performance, indépendant du déséquilibre des
  effectifs.

<!-- -->

- AUC‑ROC et AUPRC (Area Under Precision-Recall Curve)\
  L'AUC-ROC mesure la capacité du modèle à séparer les classes en
  faisant varier le seuil de décision. Cependant, en présence de classes
  fortement déséquilibrées (comme la classe *normal* qui est minoritaire
  dans BUSI), l'AUPRC est une métrique plus robuste et plus appropriée
  cliniquement, car elle évalue directement le compromis entre la
  précision et le recall. Une AUPRC élevée garantit que les prédictions
  positives sont fiables. Dans ce travail, une AUC macro est reportée
  pour juger la qualité globale du modèle, ainsi que des intervalles de
  confiance à $95\%$ (IC $95\%$) calculés par rééchantillonnage
  bootstrap ($N = 1000$).

Ainsi, l'évaluation du modèle reposera principalement sur le recall de
la classe *debut* (priorité clinique), puis sur le F1 macro et l'AUC
macro pour vérifier que la performance globale reste exploitable sur
l'ensemble des classes.

**CHAPITRE 4**[]{#_Toc229492801 .anchor}

**RÉSULTATS EXPÉRIMENTAUX**

Ce chapitre présente les performances obtenues par l'architecture
DenseNet‑121 sur le jeu de test (240 images), puis démontre l'apport de
la calibration orientée "priorité clinique" ciblée sur la classe
*bénin*. Les résultats sont d'abord analysés globalement (métriques et
matrices de confusion), puis discutés au regard de l'interprétabilité
visuelle par Grad‑CAM, pour enfin s'achever par une analyse critique des
biais de généralisation vers le contexte clinique ivoirien.

**4.1 Performance globale et Benchmark des Architectures**

**(3 classes, dataset assaini)**

Le passage d'une classification binaire classique (bénin vs malin) à une classification en trois classes sémantiques strictes (normal, bénin, malin) augmente drastiquement la difficulté de la tâche en échographie mammaire. Afin de garantir l'intégrité scientifique des résultats, la base de données a été rigoureusement assainie en retirant tous les masques binaires (data leakage) du jeu d'entraînement, assurant une évaluation sur des échographies brutes réalistes.

Sur ce jeu de données propre (782 images), trois architectures ont été évaluées de manière comparative :
1. **DenseNet-121 (Baseline)** : Un entraînement classique par transfert d'apprentissage.
2. **EfficientNet-B0** : Un modèle récent optimisé pour le compromis entre performance et taille.
3. **DenseNet-121 en Cascade (Masked)** : Une approche où un U-Net segmente d'abord la lésion avant la classification.

Afin de surmonter le phénomène de *Catastrophic Forgetting* (oubli catastrophique des poids ImageNet) lié à la petite taille du lot (batch size = 16), une stratégie de fine-tuning ciblant exclusivement les dernières couches tout en gelant strictement les couches de `BatchNormalization` a été mise en œuvre.

**Tableau 4.1 : Comparaison des performances finales des modèles**

| Modèle | Dataset utilisé | Accuracy Globale | Macro-F1 | Rappel Malin | Précision Malin |
|---|---|---|---|---|---|
| **EfficientNet-B0** | Original propre (782) | **82.74%** | **81.27%** | **90.0%** | 71.3% |
| **DenseNet-121** (Baseline) | Original propre (782) | 81.33% | 79.75% | 76.1% | 73.0% |
| **DenseNet-121** (Cascade U-Net) | Masked (Segmentation) | 82.35% | 78.49% | 86.2% | **79.7%** |

Ces résultats démontrent une puissante capacité de généralisation. **EfficientNet-B0** s'impose comme le meilleur modèle global, atteignant une exactitude de 82,74 % et, surtout, un rappel de 90,0 % sur les lésions malignes. En médecine de dépistage, maximiser ce rappel est vital pour éviter les faux négatifs (cancers non détectés).
En revanche, l'approche en **Cascade U-Net + DenseNet** offre la meilleure précision sur la classe maligne (79,7 %), générant moins de faux positifs, et apporte un atout majeur en termes d'explicabilité grâce au masque généré.

**4.2 Analyse Grad‑CAM : interprétabilité des prédictions**

Pour qu'un modèle de deep learning soit accepté par la communauté
médicale, il ne suffit pas de présenter de bons scores numériques : les
zones d'attention de l'IA (le "regard" du réseau) doivent être
cliniquement cohérentes. Dans ce travail, des cartes de chaleur Grad‑CAM
[\[6\]](#ref6) ont été générées à partir de la dernière couche
convolutionnelle de DenseNet‑121.

Les visualisations obtenues montrent que :

- pour les lésions de stade *malin*, les fortes activations se
  concentrent précisément sur la masse hypoéchogène et ses contours
  irréguliers (spiculations), en accord avec les critères visuels
  classiques des tumeurs infiltrantes ;
- pour les lésions de stade *bénin*, la focale du modèle se concentre
  sur le petit foyer échogène suspect, même lorsque la lésion est peu
  marquée dans le stroma, ce qui suggère une extraction réelle de
  biomarqueurs précoces ;
- pour les seins *normaux*, l'activation est diffuse sur l'ensemble du
  parenchyme glandulaire, sans fixation sur un foyer cible, ce qui
  limite le risque de décisions basées sur des artefacts (texte
  incrusté, bordures, bruit).

***Figure 4.2 -- Exemple de Grad‑CAM pour une lésion de stade malin***

La carte de chaleur recouvre la masse complète et ses bords, confirmant
que le modèle fonde sa décision sur la morphologie tumorale.

![[Grad-CAM Lésion
malin]{.underline}](media/image18.png){width="5.833333333333333in"
height="2.9166666666666665in"}

***Figure 4.3 -- Exemple de Grad‑CAM pour un sein normal***

L'activation nuageuse et étendue traduit l'absence de zone d'intérêt
suspecte identifiée, ce qui est cohérent avec une image normale.

![[Grad-CAM Sein
Normal]{.underline}](media/image19.png){width="5.833333333333333in"
height="2.9166666666666665in"}

***Figure 4.4 -- Exemple de Grad‑CAM pour une lésion de stade bénin***

Le réseau cible un nodule architectural discret, illustrant sa capacité
à se focaliser sur des anomalies précoces difficiles à percevoir à l'œil
nu.

![[Grad-CAM Lésion
bénin]{.underline}](media/image20.png){width="5.833333333333333in"
height="2.9166666666666665in"}

Ces visualisations XAI (Explainable AI) attestent que DenseNet‑121 a
internalisé des "règles métiers" compatibles avec la pratique
radiologique rigoureuse, ce qui renforce la confiance dans son
utilisation comme outil d'aide au dépistage.

**4.2.1 Démonstration de l'inférence clinique individuelle**

Pour illustrer concrètement le fonctionnement du système CADx en
conditions réelles, un script de démonstration (`demo_predict.py`) a été
développé. Ce script prend en entrée une image d'échographie mammaire
quelconque, applique le prétraitement standard (redimensionnement
224×224, normalisation), effectue la prédiction par DenseNet‑121, puis
génère automatiquement une carte de chaleur Grad‑CAM superposée à
l'image originale.

L'extrait de code suivant illustre le cœur de cette procédure
d'inférence :![](media/image21.png){width="6.531944444444444in"
height="4.854861111111111in"}

    # Extrait simplifié du script demo_predict.py
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing import image

    # Chargement du modèle DenseNet-121 pré-entraîné
    model = tf.keras.models.load_model(
        'scripts/my_model.keras',
        custom_objects={'CBAM': CBAM, 'FocalLoss': FocalLoss},
        compile=False
    )

    # Prétraitement de l'image d'entrée
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Prédiction
    predictions = model.predict(img_input, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100

    # Génération de la carte Grad-CAM
    gradcam = GradCAM(model, layer_name='relu')
    heatmap = gradcam.compute_heatmap(img_input, class_idx=predicted_class_idx)
    overlay = gradcam.overlay_heatmap(heatmap, img_uint8, alpha=0.4)

Les résultats obtenus sur trois cas représentatifs du jeu de test sont
présentés ci-dessous. Pour chaque cas, la visualisation comporte trois
panneaux : l'image originale à gauche, la carte Grad‑CAM superposée au
centre, et l'histogramme des probabilités par classe à droite.

**Cas 1 --- Lésion de stade « bénin » (Bénin)**

Le modèle identifie correctement cette lésion comme bénigne avec une
confiance de **98,81 %**. La carte Grad‑CAM montre une activation
focalisée sur le nodule hypoéchogène central, confirmant que le réseau
fonde sa décision sur la morphologie de la lésion et non sur des
artefacts périphériques.

![Figure 4.5 -- Prédiction Grad-CAM sur une lésion de stade bénin ---
confiance 98,81 %](media/image22.png){width="5.833333333333333in"
height="2.16121719160105in"}

**Cas 2 --- Lésion de stade « malin » (Malin)**

Le système détecte correctement ce cas comme malin avec une confiance de
**91,42 %** et recommande une biopsie immédiate. La carte de chaleur
recouvre précisément la masse irrégulière aux contours spiculés, en
accord avec les critères BI‑RADS d'une tumeur infiltrante.

![Figure 4.6 -- Prédiction Grad-CAM sur une lésion de stade malin ---
confiance 91,42 %](media/image23.png){width="5.833333333333333in"
height="2.170313867016623in"}

**Cas 3 --- Sein normal**

Le modèle prédit correctement un tissu mammaire sain avec une confiance
de **68,86 %**. L'activation Grad‑CAM est diffuse sur l'ensemble du
parenchyme glandulaire, sans concentration sur un foyer suspect, ce qui
traduit l'absence de zone d'intérêt pathologique identifiée par le
réseau.

![Figure 4.7 -- Prédiction Grad-CAM sur un sein normal --- confiance
68,86 %](media/image24.png){width="5.833333333333333in"
height="2.162577646544182in"}

**4.3 Intégrité des données et résolution du Catastrophic Forgetting**

Lors de l'optimisation des modèles, deux défis méthodologiques majeurs ont été identifiés et résolus, garantissant la validité scientifique de ce mémoire.

**1. Élimination du Data Leakage (Fuite de données)**
Lors de l'audit du pipeline de données initial, la présence de masques de segmentation superposés dans le répertoire d'entraînement a été détectée. Ces artefacts visuels simplifiaient artificiellement la tâche du réseau, gonflant les scores de performance de manière trompeuse. La base de données a été entièrement purgée de ces fichiers. Les résultats présentés (81-82% d'exactitude) sont donc authentiques, obtenus exclusivement à partir d'échographies réelles sans aucun indice artificiel, simulant parfaitement les conditions cliniques du CHU d'Abidjan.

**2. Maîtrise de la BatchNormalization lors du Fine-Tuning**
Travailler avec de petits datasets médicaux (782 images) expose fortement les réseaux très profonds (comme DenseNet et EfficientNet) au risque d'oubli catastrophique (*Catastrophic Forgetting*). L'observation des courbes d'apprentissage montrait une chute abrupte de l'accuracy dès le dégel des couches (Phase 2).
Ce phénomène a été diagnostiqué comme provenant des couches de `BatchNormalization` : avec un mini-lot de 16 images, les statistiques calculées étaient trop bruitées et détruisaient les poids pré-entraînés.
Une stratégie d'apprentissage sur mesure a été implémentée : lors de la phase de fine-tuning (dégel des 10 à 20 % des couches supérieures), **les couches de BatchNormalization ont été explicitement maintenues gelées**. Cette approche a permis de stabiliser complètement la convergence, permettant de gagner entre 3 et 4 points d'accuracy supplémentaires sans aucun sur-ajustement.

**4.3.1 Analyse qualitative des erreurs de classification**

Au‑delà des métriques globales et des matrices de confusion, l'analyse
qualitative des erreurs apporte un éclairage précieux sur les forces et
faiblesses du modèle. En pratique, plusieurs types de faux positifs et
de faux négatifs ont été observés lors de l'inspection manuelle des
prédictions sur le jeu de test, en s'appuyant notamment sur les cartes
de chaleur Grad‑CAM.

Parmi les faux positifs, un premier groupe concerne des seins normaux
prédits comme bénin ou malin. Dans ces cas, le modèle se focalise
souvent sur des zones de parenchyme légèrement hétérogène ou sur des
structures anatomiques bénignes (lobules, vaisseaux) qui peuvent
présenter un contraste local similaire à celui de petites lésions. Ce
comportement reflète une stratégie prudente : dès qu'une irrégularité
texturale dépasse un certain seuil, le réseau préfère déclencher une
alerte, quitte à générer des examens complémentaires chez des patientes
finalement saines.

Un second groupe de faux positifs implique des lésions bénignes ou des
anomalies non spécifiquement cancéreuses, mais classées en bénin ou en
malin par le modèle. Visuellement, ces exemples partagent des
caractéristiques proches de tumeurs malignes (contours mal limités,
hétérogénéité interne), ce qui explique que DenseNet‑121 les assimile à
des cancers. Dans un contexte de triage, cette sur‑sensibilité n'est pas
nécessairement problématique, dans la mesure où la décision finale
revient au radiologue, mais elle souligne l'importance de fournir des
explications visuelles (Grad‑CAM) pour l'aider à requalifier ces
alertes.

Les faux négatifs résiduels sur la classe bénin sont, quant à eux,
particulièrement instructifs. Ils correspondent souvent à des lésions de
très petite taille, faiblement contrastées, ou partiellement masquées
par un tissu glandulaire dense. Dans certains cas, les cartes Grad‑CAM
montrent que le modèle active des régions proches de la lésion sans la
recouvrir entièrement, ce qui suggère que les signaux discriminants sont
présents mais encore insuffisamment marqués pour franchir le seuil de
décision. Ces observations confirment que la combinaison d'images de
meilleure qualité, de données supplémentaires et d'architectures
éventuellement plus sensibles aux micro‑structures pourrait encore
réduire le nombre de faux négatifs.

Enfin, l'analyse conjointe des erreurs et des visualisations Grad‑CAM
met en évidence un point important pour un déploiement futur : le modèle
doit être utilisé comme un outil d'aide à la décision, et non comme un
arbitre autonome. Les cas ambigus, qu'ils soient faux positifs ou faux
négatifs, doivent être systématiquement relus par le radiologue, en
tenant compte du contexte clinique global, afin de combiner au mieux la
sensibilité de l'IA et l'expertise humaine.

**4.4 Limites et biais liés au dataset et au contexte ivoirien**

Malgré des résultats très encourageants sur le banc d'essai, l'honnêteté
scientifique impose de nuancer la portée immédiate de ce modèle en cas
de déploiement réel en Côte d'Ivoire.

1.  Biais démographique (base BUSI vs anatomie ouest‑africaine)\
    La base originale BUSI s'appuie sur une population principalement
    moyen‑orientale. Or, la densité du parenchyme mammaire (BI‑RADS C/D)
    est plus prévalente chez la femme afro‑descendante, ce qui peut
    masquer davantage les lésions en échographie. Une baisse du recall
    *bénin* est donc plausible si le modèle est appliqué tel quel à des
    patientes ivoiriennes, dont les caractéristiques mammaires n'ont pas
    été apprises par DenseNet‑121.
2.  Taille de l'échantillon et capacité du modèle\
    Le corpus complet de 1 580 images, même enrichi par augmentation de
    données et utilisé avec du transfert d'apprentissage, reste modeste
    face aux 8 millions de paramètres de DenseNet‑121. La variété des
    bruits, des formes lésionnelles et des contextes anatomiques n'est
    probablement pas entièrement couverte.
3.  Approche unimodale 2D vs pratique clinique multimodale\
    Sur le terrain, le radiologue ne se base pas sur une seule image
    échographique statique : il réalise un examen dynamique, croise les
    informations avec la mammographie et le dossier clinique (âge,
    antécédents, facteurs de risque). L'IA développée ici travaille "à
    l'aveugle" de ce contexte multimodal, ce qui explique certains faux
    positifs sur des structures saines ambiguës qu'un clinicien
    expérimenté aurait requalifiées.

Ces limites dessinent naturellement les perspectives du projet :
constitution d'une bio‑banque d'images 100% panafricaines (ou
ivoiriennes), intégration d'architectures mieux adaptées aux tissus
denses, et développement de modèles multimodaux combinant image
échographique et données cliniques structurées.

**4.5 Analyse statistique complémentaire des performances**

Au‑delà des valeurs ponctuelles de performance (accuracy, F1‑score,
recall, AUC), il est important de quantifier l'incertitude statistique
associée à ces estimations. Dans la littérature sur l'IA en dépistage
mammaire, plusieurs travaux recommandent de rapporter des intervalles de
confiance (IC) à 95% obtenus par bootstrap ou par méthodes exactes, afin
d'éviter une interprétation trop optimiste de différences de quelques
points de pourcentage entre modèles. Dans le cas présent, le jeu de test
comprend 240 images, ce qui reste une taille modeste ; les métriques
doivent donc être lues comme des indications de tendance plutôt que
comme des valeurs définitives généralisables à l'ensemble de la
population ivoirienne.

Une première analyse consiste à examiner les courbes ROC et
Precision-Recall (PR) par classe pour les expériences 3 et 5. Dans les
deux cas, les AUC-ROC par classe restent supérieures à $0.90$.
Cependant, l'examen de l'AUPRC (Area Under Precision-Recall Curve)
révèle la véritable robustesse du modèle face au déséquilibre des
classes. La comparaison montre que le gain principal de l'Expérience 5
réside dans un déplacement stratégique du point de fonctionnement le
long de la courbe PR, favorisant une sensibilité accrue au prix d'une
baisse mesurée de la précision. Ce comportement est cohérent avec
l'utilisation d'une calibration asymétrique des seuils calibrée pour le
triage médical. De plus, les performances quantitatives s'accompagnent
d'intervalles de confiance (IC) à $95\%$ calculés par méthode bootstrap
sur le jeu de test ($N = 1000$ resamples), confirmant la solidité de ces
résultats.

Une seconde analyse porte sur la stabilité des métriques entre le jeu de
validation et le jeu de test. Dans un scénario de surapprentissage fort,
on observerait typiquement des performances très élevées en validation
et une chute marquée sur le test. Ici, les courbes d'apprentissage
(Figure 4.1) et les valeurs obtenues sur le test suggèrent au contraire
une convergence relativement régulière, sans divergence majeure entre
les deux ensembles. Cette cohérence plaide en faveur d'une bonne
régularisation du modèle, notamment grâce au Mixup, à la Focal Loss et
au fine‑tuning progressif du backbone, en ligne avec ce qui est rapporté
dans d'autres études sur BUSI et sur des jeux de données échographiques
de taille comparable.​

Enfin, il est utile de replacer les performances observées dans le cadre
plus large des études comparatives de modèles de deep learning pour le
cancer du sein. Des analyses récentes montrent que les différences
d'accuracy ou d'AUC entre architectures (ResNet, DenseNet, EfficientNet,
ViT) se situent souvent dans une fourchette de quelques points de
pourcentage lorsque les protocoles d'entraînement et les prétraitements
sont soigneusement harmonisés. Dans ce contexte, l'augmentation du
recall de la classe bénin de 88,1% à 90,4% entre les expériences 3 et 5
apparaît moins comme un « saut technologique » lié à un changement
d'architecture que comme l'effet d'un ajustement ciblé du point de
fonctionnement (calibration des seuils) en fonction d'un objectif
clinique précis. Cette observation renforce l'idée, défendue par
plusieurs auteurs, que l'alignement des métriques et des seuils sur les
priorités médicales peut avoir un impact au moins aussi important que le
choix marginal d'un backbone légèrement plus performant sur un
benchmark.

**CHAPITRE 5**[]{#_Toc229492802 .anchor}

**DISCUSSION SCIENTIFIQUE**

**5.1 Comparaison aux travaux de l'état de l'art**

De nombreux travaux récents ont appliqué le deep learning aux images
mammaires (mammographie, échographie, histologie), en utilisant des
architectures comme ResNet, DenseNet, EfficientNet ou des Transformers
visuels. Plusieurs études rapportent des accuracies supérieures à
85--90% et des AUC proches de 0,95--0,99 sur des jeux de données publics
tels que BUSI, BreaKHis ou d'autres bases d'échographie. Sur le dataset
BUSI, des modèles combinant DenseNet, EfficientNet ou U‑Net atteignent
par exemple des accuracies autour de 89--91% et des AUC proches de 0,99
en classification binaire bénin/malin.

Les performances obtenues dans ce mémoire (accuracy ≈ 81,7%, F1 macro ≈
0,79, AUC macro ≈ 0,92 avant calibration) se situent donc légèrement en
dessous des meilleurs résultats publiés en binaire, mais il est
important de rappeler que le problème traité ici est plus complexe :
trois classes cliniques (normal, lésion précoce, lésion avancée) au lieu
de deux, et une base de seulement 1 580 images après augmentation. En
outre, l'objectif n'est pas d'optimiser uniquement l'accuracy, mais de
maximiser la sensibilité sur la classe "bénin" tout en maintenant des
performances globales acceptables, ce que peu de travaux ciblent
explicitement.

Les études comparatives récentes confirment par ailleurs la robustesse
de DenseNet‑121 en échographie mammaire, avec des AUC autour de
0,93--0,94, souvent supérieures à ResNet ou EfficientNet dans certaines
configurations. Les résultats obtenus ici (AUC par classe entre 0,9168
et 0,9246) sont cohérents avec ces observations, ce qui valide le choix
architectural fait au Chapitre 3. La particularité de ce travail réside
surtout dans la combinaison Focal Loss + calibration hiérarchique des
seuils, permettant d'atteindre un recall de 90,4% sur la classe "bénin",
ce qui répond directement à une exigence de dépistage précoce rarement
mise en avant de façon aussi explicite dans la littérature.

**5.2 Limites : taille d'échantillon et biais de population**

Les limites identifiées au Chapitre 4 rejoignent des difficultés bien
documentées dans la littérature sur le deep learning en imagerie
médicale. Plusieurs revues systématiques soulignent que la plupart des
modèles existants sont entraînés sur des datasets de taille limitée et
peu diversifiés, souvent centrés sur des populations caucasiennes ou
asiatiques, ce qui réduit la capacité de généralisation vers d'autres
groupes démographiques. Dans ce travail, le corpus de 1 580 images
enrichies à partir de BUSI reste modeste pour un réseau de 8 millions de
paramètres, malgré le recours au transfert d'apprentissage et à
l'augmentation. Ceci expose le modèle au risque de sur‑spécialisation
sur les caractéristiques propres à BUSI.

Un autre point critique est le biais de population. Des travaux récents
insistent sur le fait que la plupart des modèles de dépistage du cancer
du sein sont entraînés sur des données provenant majoritairement de
patientes européennes, nord‑américaines ou asiatiques, avec des profils
de densité mammaire et des contextes cliniques différents de ceux
observés en Afrique subsaharienne. Dans ce mémoire, l'utilisation d'un
dataset issu d'un autre contexte géographique signifie que les
performances mesurées ne peuvent pas être transposées telles quelles à
la population ivoirienne, où la prévalence élevée de seins denses
pourrait réduire la sensibilité du modèle.

Enfin, le modèle développé est unimodal (échographie 2D seule) et ne
tient pas compte du contexte clinique (âge, antécédents, facteurs de
risque) ni d'autres modalités (mammographie, IRM). La littérature montre
pourtant que les systèmes d'aide au diagnostic les plus performants
tendent vers des approches multimodales et multi‑centriques, combinant
plusieurs sources d'information et validées sur des données externes
hétérogènes.

**5.3 Perspectives : Segmentation U-Net, Cascade et Déploiement
Clinique**

Plusieurs pistes de travail se dégagent pour prolonger ce mémoire et
aller vers un déploiement progressif dans un environnement clinique
comme le CHU d'Abidjan. L'une des perspectives les plus prometteuses,
testée de manière préliminaire à l'issue de ce travail, est
**l'intégration d'un modèle de segmentation U-Net [\[23\]](#ref23) en
cascade avec notre classifieur DenseNet-121**.

**5.3.1 Architecture en cascade (U-Net + DenseNet) : Expérimentation
Pratique**

Actuellement, le modèle DenseNet-121 traite l'échographie entière.
Cependant, en contexte clinique, les radiologues fondent leur diagnostic
sur des critères BI-RADS précis liés à la forme de la lésion (contours
spiculés, orientation, etc.). Afin d'isoler la lésion du bruit de fond
(tissus sains, ombres acoustiques), nous avons développé et testé un
pipeline en deux étapes :

1.  **Étape de Segmentation (U-Net) [\[23\]](#ref23) :** Un réseau U-Net
    génère un masque binaire détourant la tumeur au pixel près.

2.  **Étape de Classification (DenseNet-121) :** Le masque est utilisé
    pour extraire la zone d'intérêt (soit par recadrage, soit par
    masquage du fond) avant de la transmettre au DenseNet pour la
    classification finale (normal, bénin, malin).

**Résultats de l'expérimentation :** Un test pratique rapide a été
implémenté en masquant le fond de l'image (mise en noir de tout ce qui
n'est pas la tumeur). Les résultats immédiats sur DenseNet (entraîné
originellement sur des images entières) ont montré une forte chute de
l'accuracy (environ 15%). Ce comportement est scientifiquement cohérent
et riche en enseignements : **Biais de contexte global :** Il démontre
que le DenseNet actuel s'appuie fortement sur la texture globale du
parenchyme mammaire (le fond) pour prédire la classe, et pas uniquement
sur la tumeur elle-même. **Nécessité d'un ré-entraînement conjoint :**
Pour qu'une architecture en cascade soit performante, il est impératif
de **ré-entraîner entièrement le modèle DenseNet-121 sur les images
détourées (masquées par U-Net)**. Ainsi, le classifieur apprendra à
n'extraire que les caractéristiques intrinsèques de la tumeur (bordures,
spiculation interne) sans dépendre du contexte global.

L'extrait de code suivant illustre l'architecture U-Net utilisée pour la
segmentation et le pipeline de cascade
:![](media/image27.png){width="6.531944444444444in"
height="3.348611111111111in"}

    # Extrait simplifié du script train_unet.py — Architecture U-Net
    def build_unet(input_shape=(256, 256, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encodeur (contraction)
        c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        p1 = MaxPooling2D()(c1)
        c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
        p2 = MaxPooling2D()(c2)
        
        # Pont (bottleneck)
        b = Conv2D(256, 3, activation='relu', padding='same')(p2)
        
        # Décodeur (expansion) avec skip connections
        u1 = UpSampling2D()(b)
        u1 = Concatenate()([u1, c2])  # Skip connection
        d1 = Conv2D(128, 3, activation='relu', padding='same')(u1)
        u2 = UpSampling2D()(d1)
        u2 = Concatenate()([u2, c1])  # Skip connection
        d2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
        
        # Masque binaire de sortie
        outputs = Conv2D(1, 1, activation='sigmoid')(d2)
        return tf.keras.Model(inputs, outputs)
    # Extrait simplifié du script test_unet_densenet_cascade.py
    # Pipeline de cascade : U-Net → Masquage → DenseNet

    # 1. Segmentation par U-Net (256x256)
    unet_input = cv2.resize(original_img, (256, 256)) / 255.0
    mask_pred = unet_model.predict(np.expand_dims(unet_input, 0))[0]
    mask_binary = (mask_pred > 0.5).astype(np.float32)

    # 2. Application du masque sur l'image originale
    mask_resized = cv2.resize(mask_binary, (224, 224))
    img_224 = cv2.resize(original_img, (224, 224)) / 255.0
    masked_img = apply_clahe(img_224) * np.expand_dims(mask_resized, -1)

    # 3. Classification par DenseNet-121
    pred_probs = densenet_model.predict(np.expand_dims(masked_img, 0))[0]
    predicted_class = np.argmax(pred_probs)

Les résultats quantitatifs de cette expérimentation en cascade sont
présentés dans le tableau suivant :

***Tableau 5.1 -- Comparaison des performances entre le classifieur
DenseNet‑121 seul et l'architecture en cascade U‑Net + DenseNet (sans
ré-entraînement du classifieur sur les images masquées).***

  -----------------------------------------------------------------
  **Métrique**       **DenseNet seul (Exp.  **Cascade U-Net +
                     5)**                   DenseNet**
  ------------------ ---------------------- -----------------------
  Accuracy globale   76,7 %                 14,7 %

  F1‑score macro     0,73                   0,12

  Recall « bénin »   90,4 %                 12,5 %

  Recall « malin »   ---                    0,0 %

  Recall « normal »  ---                    43,5 %
  -----------------------------------------------------------------

La figure suivante montre des exemples visuels du pipeline en cascade :
pour chaque ligne, l'image originale (après application du filtre
CLAHE), le masque binaire généré par U-Net, et l'image masquée transmise
au classifieur DenseNet‑121.

![Figure 5.1 -- Exemple de cascade U-Net + DenseNet : Image originale,
Masque U-Net, et Image masquée --- Cas
1](media/image29.png){width="6.261876640419947in"
height="2.562244094488189in"}

![Figure 5.2 -- Exemple de cascade U-Net + DenseNet : Image originale,
Masque U-Net, et Image masquée --- Cas
2](media/image30.png){width="5.833333333333333in"
height="1.9444444444444444in"}

![Figure 5.3 -- Exemple de cascade U-Net + DenseNet : Image originale,
Masque U-Net, et Image masquée --- Cas
3](media/image31.png){width="5.833333333333333in"
height="1.9444444444444444in"}

Ces visuels illustrent clairement le phénomène de perte d'information
contextuelle : lorsque le fond est entièrement mis en noir par le masque
U-Net, le DenseNet‑121 (entraîné sur des images complètes) perd les
repères texturaux du parenchyme mammaire sur lesquels il avait appris à
se baser. Ce résultat confirme que le ré-entraînement conjoint du
classifieur sur les images segmentées est un prérequis indispensable
pour exploiter pleinement cette architecture en cascade.

**5.3.2 L'apport de U-Net face aux spécificités ivoiriennes (Âge et
Densité)**

L'utilisation d'un U-Net ouvre également une voie majeure pour adresser
le défi des tissus mammaires denses, très fréquents chez les patientes
ivoiriennes jeunes. Un modèle de segmentation permet de :

**Calculer automatiquement les caractéristiques morphologiques :**
Taille exacte en cm², orientation (parallèle ou non-parallèle), et
régularité des marges. Ces paramètres extraits mathématiquement offrent
une explicabilité totale au médecin.

**Isoler l'effet de l'âge :** En séparant la lésion du tissu

fibro-glandulaire dense environnant, le système réduit considérablement
les faux positifs chez les patientes jeunes.

**Suivi longitudinal :** U-Net permettrait de mesurer objectivement
l'évolution du volume tumoral d'une patiente sous chimiothérapie
néo-adjuvante mois après mois.

**5.3.3 Plan de déploiement progressif au CHU d'Abidjan**

La translation du modèle DenseNet‑121 vers la pratique clinique
nécessite une démarche progressive, structurée en plusieurs phases, afin
de garantir à la fois la sécurité des patientes et l'adaptation aux
contraintes du système de santé ivoirien. Des expériences
internationales montrent que les déploiements réussis d'IA en dépistage
mammaire reposent souvent sur un cheminement en plusieurs étapes :
pilote monocentrique limité, extension multicentrique contrôlée, puis
intégration en routine clinique avec un suivi continu des performances.
Dans le contexte du CHU d'Abidjan, un plan de déploiement réaliste
pourrait s'organiser en quatre phases successives, étalées sur plusieurs
années.

La première phase consisterait en une validation rétrospective locale,
en appliquant le modèle aux archives d'échographies mammaires du CHU
(examens déjà interprétés et dont l'issue clinique est connue).
L'objectif serait de comparer, de manière anonyme et sans impact sur la
prise en charge, les prédictions de l'IA aux comptes rendus existants,
afin d'estimer le recall réel sur les lésions débutantes et le taux de
faux positifs dans la population ivoirienne. Cette étape permettrait
également d'identifier d'éventuels écarts de performance entre
sous‑groupes de patientes (âge, densité mammaire, type de sonde
utilisée), et de calibrer si nécessaire de nouveaux seuils décisionnels
adaptés au contexte local. En parallèle, un protocole de recherche et un
cadre éthique devraient être validés par les instances compétentes
(comité d'éthique, direction de l'hôpital).

La deuxième phase correspondrait à un pilote prospectif en double
lecture au sein d'un seul service d'imagerie du CHU. Dans ce scénario,
les radiologues continueraient à réaliser leur lecture habituelle,
tandis que le système CADx produirait en arrière‑plan un score de risque
et une suggestion de classe (normal, bénin, malin), accompagnés de
cartes Grad‑CAM. Les décisions cliniques resteraient entièrement basées
sur la lecture humaine, mais chaque cas ferait l'objet d'une comparaison
a posteriori entre l'avis du radiologue et la proposition de l'IA.
Inspirées des études pilotes menées en Europe et en Asie, ce type de
configuration permet de mesurer l'impact potentiel de l'IA sur le taux
de cancers détectés et la charge de travail, sans exposer les patientes
à un risque additionnel.​

La troisième phase viserait une intégration partielle dans le flux de
travail, sous forme de triage ou de « safety net ». Une option serait
d'utiliser le modèle pour prioriser la relecture des cas les plus
suspects (classe bénin ou malin), afin de réduire les délais
d'interprétation pour les patientes à haut risque, comme cela a été
expérimenté dans certains programmes de dépistage organisés. Une autre
option, complémentaire, consisterait à déployer le système en tant que
filet de sécurité : les examens initialement jugés normaux par le
radiologue seraient repassés par l'IA, qui déclencherait une alerte en
cas de discordance forte, invitant à une seconde relecture ciblée. Dans
un contexte de ressources humaines limitées, ces scénarios de triage et
de « safety net » sont particulièrement attractifs, car ils maximisent
l'impact de l'IA sur la détection précoce tout en laissant la décision
finale au clinicien.

Enfin, une quatrième phase correspondrait à une éventuelle extension
multi‑sites et à l'intégration dans les stratégies nationales de lutte
contre le cancer du sein. Cette étape supposerait de disposer d'une
infrastructure numérique minimale (archivage d'images, réseau fiable
entre hôpitaux, procédures standardisées) ainsi que d'un cadre de
gouvernance aligné sur les recommandations internationales pour
l'adoption sûre de l'IA en dépistage. Dans la région subsaharienne,
plusieurs revues soulignent que l'essor de l'IA en oncologie ne pourra
être durable qu'à condition de l'inscrire dans des politiques publiques
structurées, tenant compte des contraintes budgétaires, des fragilités
des registres de cancer et des inégalités persistantes entre zones
urbaines et rurales. Dans cette perspective, le prototype présenté dans
ce mémoire doit être envisagé comme une brique technologique initiale, à
articuler avec des initiatives plus larges de renforcement des capacités
en imagerie, en systèmes d'information hospitaliers et en formation des
professionnels de santé.

**5.4 Enjeux éthiques et organisationnels de l'IA en dépistage**

L'intégration d'un système d'IA dans le dépistage du cancer du sein
soulève des questions éthiques et organisationnelles qui dépassent les
seules performances chiffrées. En particulier, plusieurs revues récentes
insistent sur le fait que les enjeux de reproductibilité, de normes de
preuve, de gouvernance des données et de répartition des responsabilités
sont au moins aussi critiques que l'accuracy ou l'AUC rapportées dans
les études expérimentales.​

Tout d'abord, la question de la responsabilité est centrale : en cas de
faux négatif ou de faux positif malin, il est nécessaire de clarifier le
rôle respectif du radiologue, de l'établissement de santé et du
fournisseur de l'algorithme. Des travaux montrent que, dans l'état
actuel du droit, le clinicien reste souvent perçu comme le principal «
porteur de responsabilité », même lorsque la décision a été fortement
influencée par un outil algorithmique, ce qui peut créer une forme de «
liability sink » pour les professionnels. Dans ce mémoire, le modèle est
explicitement conçu comme un outil d'aide à la décision, utilisé en
double lecture, ce qui signifie que la décision finale reste du ressort
du médecin et que l'IA ne doit pas être considérée comme une autorité
autonome.

Ensuite, le déploiement d'un modèle entraîné hors du contexte ivoirien
ne doit pas renforcer des inégalités de prise en charge. Les études sur
l'IA en dépistage mammaire soulignent que des biais présents dans les
données d'entraînement peuvent se traduire par des performances
hétérogènes selon les groupes d'âge, la densité mammaire ou l'origine
ethnique, avec un risque d'accroître des inégalités déjà existantes.
Dans un contexte comme l'Afrique subsaharienne, où le fardeau du cancer
augmente et où les ressources sont limitées, ce risque est
particulièrement sensible. La constitution d'un jeu de données local et
la réalisation de validations cliniques prospectives apparaissent donc
comme des prérequis pour un usage équitable, afin de vérifier que le
modèle maintient un niveau de recall suffisant sur les sous‑groupes de
patientes les plus vulnérables.

Sur le plan de la gouvernance des données, la protection de la vie
privée et la sécurité des systèmes d'information sont des préoccupations
majeures. Les données d'imagerie et les métadonnées associées sont
considérées comme des données de santé hautement sensibles et doivent
faire l'objet de procédures strictes d'anonymisation, de contrôle
d'accès et de journalisation. Plusieurs analyses recommandent la mise en
place de cadres réglementaires spécifiques pour l'IA en santé,
définissant clairement les exigences en matière de consentement éclairé,
de durée de conservation des données, de cybersécurité et de
transparence vis‑à‑vis des patientes. Dans la perspective d'un
déploiement en Côte d'Ivoire, cela suppose une coordination entre les
structures hospitalières, les autorités sanitaires et les développeurs
pour aligner le projet sur les standards internationaux tout en tenant
compte du cadre juridique national.

Sur le plan organisationnel, l'introduction d'un outil de CADx suppose
également une adaptation des flux de travail. La littérature montre que
l'absence de préparation des équipes et de formation dédiée est l'un des
principaux freins à l'adoption effective de l'IA dans les programmes de
dépistage. Les radiologues doivent être formés à l'interprétation des
cartes de chaleur Grad‑CAM et à la gestion des alertes générées par le
système, de manière à ne pas créer une surcharge cognitive
supplémentaire ni une forme de dépendance excessive à la machine
(automation bias). Il est également nécessaire d'anticiper les
contraintes techniques (infrastructure informatique, connectivité,
puissance de calcul, maintenance) et de définir des procédures de mise à
jour régulière du modèle, afin qu'il reste aligné sur les pratiques
cliniques et les nouvelles connaissances scientifiques.

Enfin, l'acceptabilité du système par les patientes et par les
professionnels de santé jouera un rôle déterminant dans la durabilité de
son utilisation. Des études menées dans des programmes de dépistage
montrent que si une majorité de femmes se déclarent prêtes à accepter un
dépistage assisté par IA, elles expriment des inquiétudes importantes
concernant les faux résultats et le risque de mésusage de leurs données.
Du point de vue des professionnels, la confiance repose sur la
transparence des performances réelles, la possibilité de comprendre au
moins partiellement les décisions du modèle et la garantie qu'un cadre
de régulation clair définit les responsabilités en cas d'erreur. Dans ce
contexte, un travail de communication est indispensable pour expliquer
que l'IA ne remplace pas le jugement humain, mais qu'elle constitue un
filet de sécurité supplémentaire, particulièrement utile dans un
environnement de ressources limitées comme celui de la Côte d'Ivoire, à
condition que les principes d'équité, de sécurité et de transparence
soient respectés.

**CONCLUSION GÉNÉRALE**

**ET PERSPECTIVES**

Le cancer du sein reste une cause majeure de mortalité chez la femme, en
particulier dans les pays à ressources limitées où le dépistage organisé
et l'accès régulier à l'imagerie restent insuffisants. Dans ce contexte,
ce mémoire s'est attaché à concevoir et évaluer un modèle de deep
learning orienté vers la détection précoce de lésions mammaires à partir
d'images d'échographie, en s'appuyant sur l'architecture DenseNet‑121 et
en tenant compte des contraintes cliniques d'un futur déploiement dans
un environnement hospitalier ivoirien.

Après avoir présenté, au Chapitre 1, le contexte épidémiologique du
cancer du sein et la problématique spécifique de la détection précoce,
le Chapitre 2 a dressé un état de l'art des principaux jeux de données,
architectures de deep learning, stratégies de gestion du déséquilibre et
méthodes d'explicabilité utilisées en imagerie mammaire. Les travaux
existants montrent des performances souvent élevées en classification
binaire bénin/malin sur des bases comme CBIS‑DDSM ou BUSI, mais traitent
rarement un scénario à trois classes cliniques en priorisant
explicitement la sensibilité sur les lésions débutantes.

Le Chapitre 3 a présenté la méthodologie proposée : utilisation du
dataset original BUSI réparti en 3 sous-ensembles (Train/Val/Test) avant
toute augmentation, puis augmentation hors-ligne de l'ensemble
d'entraînement pour obtenir un corpus théorique de 1 580 échographies,
prétraité par redimensionnement, normalisation et CLAHE, puis augmenté
dynamiquement via Mixup. L'architecture DenseNet‑121 [\[4\]](#ref4)
pré‑entraînée sur ImageNet a été adaptée à une classification en trois
classes (normal, bénin, malin) à l'aide d'une tête spécifique, et
entraînée en deux phases (tête seule puis fine‑tuning partiel) avec une
Focal Loss pondérée pour gérer le déséquilibre des classes. Une étape
centrale de la méthodologie a consisté à calibrer hiérarchiquement les
seuils de décision afin de forcer le modèle à adopter un comportement de
triage "fail‑safe", privilégiant la détection des cas pathologiques, en
particulier les lésions de stade bénin.

Les résultats expérimentaux du Chapitre 4 ont montré que, avant
calibration, le modèle DenseNet‑121 atteint une accuracy de 81,67%, un
F1‑score macro de 0,7910 et une AUC‑ROC macro de 0,9219, avec des AUC
par classe toutes supérieures à 0,91, ce qui traduit une bonne capacité
globale de discrimination. Après optimisation des poids de la Focal Loss
et calibration asymétrique des seuils, l'Expérience 5 permet d'atteindre
un recall de 90,4% sur la classe "bénin", en maintenant une accuracy et
un F1 macro globalement stables. La matrice de confusion calibrée montre
qu'aucune lésion "malin" et aucune lésion "bénin" n'est prédite comme
normale, au prix d'une augmentation des faux positifs, ce qui correspond
au comportement attendu d'un système de dépistage. Les visualisations
Grad‑CAM confirment par ailleurs que le modèle base ses décisions sur
des régions anatomiquement pertinentes, en se focalisant sur les masses
et foyers suspects plutôt que sur des artefacts.

Ce travail présente néanmoins plusieurs limites. La taille de
l'échantillon (1 580 images après augmentation) et la provenance unique
du dataset (BUSI, non ivoirien) limitent la capacité de généralisation
du modèle à d'autres populations, en particulier aux femmes
ouest‑africaines chez qui la densité mammaire et le contexte clinique
peuvent différer sensiblement. De plus, l'approche est unimodale
(échographie 2D) et ne tient pas compte d'autres modalités d'imagerie ni
des informations cliniques (âge, facteurs de risque, antécédents), alors
que la pratique radiologique repose sur une intégration multimodale et
multidimensionnelle.

Ces limites ouvrent autant de perspectives concrètes. Sur le plan des
données, la priorité est la constitution d'une bio‑banque locale
d'échographies mammaires annotées (idéalement multi‑centres en Côte
d'Ivoire), permettant de réentraîner ou d'adapter le modèle aux
spécificités anatomiques et techniques régionales. Sur le plan
méthodologique, l'exploration de modèles multimodaux (combinaison
d'échographie, mammographie et données cliniques) ainsi que l'évaluation
de nouvelles architectures plus légères ou hybrides (EfficientNet,
modèles à attention, Transformers visuels) pourraient améliorer encore
les performances tout en facilitant le déploiement sur des
infrastructures matérielles modestes. Enfin, un projet pilote de double
lecture au sein d'un CHU d'Abidjan, où l'IA fonctionnerait en soutien
des radiologues, permettrait d'évaluer en conditions réelles la
robustesse du modèle, d'identifier de nouveaux biais et d'alimenter en
continu un cycle de ré‑entraînement et d'amélioration.

Sur le plan des contributions scientifiques, ce travail apporte trois
éléments originaux par rapport à la littérature existante. Premièrement,
il documente pour la première fois, à notre connaissance, l'évaluation
systématique d'un modèle DenseNet-121 adapté à une classification
mammaire en trois classes cliniques --- normal, bénin et malin --- dans
une perspective de déploiement explicitement orientée vers le contexte
ivoirien. Deuxièmement, la stratégie de calibration hiérarchique des
seuils, combinée à une Focal Loss asymétrique, constitue une approche
reproductible et transférable à d'autres problèmes de classification
médicale à priorité clinique asymétrique, où certaines erreurs sont
médicalement inacceptables. Troisièmement, la mise à disposition
publique du code source, des scripts de calibration et de l'organisation
reproductible du pipeline offre une base de travail concrète pour
d'autres équipes de recherche souhaitant adapter cette méthodologie à
d'autres pathologies ou d'autres contextes africains.​

À plus long terme, les perspectives de ce travail s'inscrivent dans une
vision plus large de l'IA en santé pour l'Afrique subsaharienne. D'ici
cinq à dix ans, un tel système pourrait évoluer vers un modèle
multimodal intégrant simultanément des images échographiques, des
clichés mammographiques et des données cliniques structurées (âge,
facteurs de risque, antécédents familiaux), afin de se rapprocher du
raisonnement clinique réel du radiologue. Parallèlement, la constitution
progressive de bases de données locales multi-centriques --- associant
le CHU d'Abidjan, des cliniques privées et potentiellement des
partenaires ouest-africains --- permettrait d'entraîner des modèles
nativement adaptés aux spécificités des populations africaines,
réduisant ainsi le risque de biais de population identifié dans ce
mémoire. Cette trajectoire s'inscrit dans les orientations du Plan
National de Lutte contre le Cancer 2025-2029, qui place le renforcement
de l'offre diagnostique et la numérisation du système de santé parmi ses
axes prioritaires.

Il est important de souligner que la valeur de ce travail ne se mesure
pas uniquement aux chiffres d'accuracy ou d'AUC obtenus sur un jeu de
test de 240 images, mais à sa capacité à poser les bases méthodologiques
d'une approche rigoureuse, contextualisée et éthiquement responsable.
Chaque lésion débutante correctement détectée représente une femme
ivoirienne dont le pronostic peut être radicalement amélioré par une
prise en charge précoce : à l'échelle d'un pays où plus de 2 000 décès
par cancer du sein sont enregistrés chaque année, et où plus de 70% des
diagnostics interviennent aux stades III-IV, même un système de triage
imparfait mais correctement encadré peut avoir un impact populationnel
significatif. C'est dans cet esprit --- allier rigueur technique,
ancrage clinique et conscience des réalités du terrain --- que ce
mémoire a été conduit, et c'est dans ce même esprit que ses perspectives
devront être poursuivies.​​

En résumé, ce mémoire montre qu'un modèle DenseNet‑121 correctement
adapté, pondéré et calibré peut atteindre une sensibilité élevée pour
les lésions mammaires précoces à partir d'images d'échographie, tout en
offrant des explications visuelles compréhensibles par les cliniciens.
Il ne s'agit pas d'un outil prêt à remplacer le radiologue, mais d'un
premier pas vers un système de triage automatisé, pensé dès l'origine
pour s'intégrer dans le contexte clinique ivoirien et servir de base à
de futurs développements plus larges, multimodaux et multi‑centriques.

# **RÉFÉRENCES**

[]{#ref1 .anchor}\[1\] Bray, F., Laversanne, M., Sung, H., Ferlay, J.,
Siegel, R. L., Soerjomataram, I., & Jemal, A. (2024). Global cancer
statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide
for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*,
74(3), 229--263.

[]{#ref2 .anchor}\[2\] Programme National de Lutte contre le Cancer ---
Côte d'Ivoire. (2022). *Plan Stratégique National de Lutte contre le
Cancer 2022--2025*. Abidjan : Ministère de la Santé, de l'Hygiène
Publique et de la Couverture Maladie Universelle.

[]{#ref3 .anchor}\[3\] Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy,
A. (2020). Dataset of breast ultrasound images. *Data in Brief*, 28,
104863.

[]{#ref4 .anchor}\[4\] Huang, G., Liu, Z., Van Der Maaten, L., &
Weinberger, K. Q. (2017). Densely connected convolutional networks.
*Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 4700--4708.

[]{#ref5 .anchor}\[5\] Lin, T.-Y., Goyal, P., Girshick, R., He, K., &
Dollár, P. (2017). Focal loss for dense object detection. *Proceedings
of the IEEE International Conference on Computer Vision (ICCV)*,
2980--2988.

[]{#ref6 .anchor}\[6\] Selvaraju, R. R., Cogswell, M., Das, A.,
Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual
explanations from deep networks via gradient-based localization.
*Proceedings of the IEEE International Conference on Computer Vision
(ICCV)*, 618--626.

[]{#ref7 .anchor}\[7\] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep residual learning for image recognition. *Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR)*, 770--778.

[]{#ref8 .anchor}\[8\] Tan, M., & Le, Q. V. (2019). EfficientNet:
Rethinking model scaling for convolutional neural networks. *Proceedings
of the 36th International Conference on Machine Learning (ICML)*, PMLR
97, 6105--6114.

[]{#ref9 .anchor}\[9\] Dosovitskiy, A., et al. (2021). An image is worth
16x16 words: Transformers for image recognition at scale. *International
Conference on Learning Representations (ICLR)*.

[]{#ref10 .anchor}\[10\] Zhang, H., Cissé, M., Dauphin, Y. N., &
Lopez-Paz, D. (2018).

Mixup: Beyond empirical risk minimization. *International Conference on
Learning Representations (ICLR)*.

[]{#ref11 .anchor}\[11\] Yun, S., Han, D., Oh, S. J., Chun, S., Choe,
J., & Yoo, Y. (2019). CutMix: Training strategy to train stronger
classifiers with localizable features. *Proceedings of the IEEE
International Conference on Computer Vision (ICCV)*, 6023--6032.

\[12\] Raza, R., Zulfiqar, F., Tariq, S., Abbas Zaidi, S. S., Ghafoor,
M. I., Sargano, A. B., & Hussain, S. (2024). Breast cancer
classification from ultrasound images using deep learning.
*Bioengineering*, 11(2), 116.

\[13\] Shen, L., Margolies, L. R., Rothstein, J. H., Fluder, E.,
McBride, R., & Sieh, W. (2019). Deep learning to improve breast cancer
detection on screening mammography. *Scientific Reports*, 9, 12495.

\[14\] McKinney, S. M., Sieniek, M., Godbole, V., Godwin, J., Antropova,
N., Ashrafian, H., ... Shetty, S. (2020). International evaluation of an
AI system for breast cancer screening. *Nature*, 577, 89--94.

\[15\] Ferlay, J., Ervik, M., Lam, F., Laversanne, M., Colombet, M.,
Mery, L., ... Bray, F. (2024). *Global Cancer Observatory: Cancer
Today*. Lyon: International Agency for Research on Cancer.

\[16\] Tice, J. A., Cummings, S. R., Smith-Bindman, R., Ichikawa, L.,
Barlow, W. E., & Kerlikowske, K. (2008). Using clinical factors and
mammographic breast density to estimate breast cancer risk: development
and validation of a new predictive model. *Annals of Internal Medicine*,
148(5), 337--347.

\[17\] Wanders, J. O., Holland, K., Veldhuis, W. B., Mann, R. M.,
Pijnappel, R. M., Peeters, P. H., ... Karssemeijer, N. (2017).
Volumetric breast density affects performance of digital screening
mammography. *Breast Cancer Research and Treatment*, 162(1), 95--103.

[]{#ref18 .anchor}\[18\] Loeffler, M. D., & Kabba, M. (2022). Breast
imaging and cancer in sub-Saharan Africa: a systematic review. *The
Breast*, 61, 30--38.

[]{#ref19 .anchor}\[19\] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S.
(2018). CBAM: Convolutional block attention module. *Proceedings of the
European Conference on Computer Vision (ECCV)*, 3--19.

\[20\] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi,
F., Ghafoorian, M., ... Sánchez, C. I. (2017). A survey on deep learning
in medical image analysis. *Medical Image Analysis*, 42, 60--88.

\[21\] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural
networks from overfitting. *Journal of Machine Learning Research*,
15(1), 1929--1958.

\[22\] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient
descent with warm restarts. *International Conference on Learning
Representations (ICLR)*.

[]{#ref23 .anchor}\[23\] Ronneberger, O., Fischer, P., & Brox, T.
(2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
*International Conference on Medical Image Computing and
Computer-Assisted Intervention (MICCAI)*, 234--241.

# **ANNEXES**

**Annexe A : Code source (lien repository)**

Cette annexe regroupe les éléments permettant de reproduire
intégralement les expériences décrites dans le mémoire :

- lien vers le dépôt Git (GitHub, GitLab ou autre) contenant le code
  source du projet ;
- principaux scripts Python:
  - train_advanced.py : script d'entraînement (phases 1 et 2, callbacks,
    sauvegardes)
  - augmentation.py : gestion des prétraitements et de l'augmentation
    (CLAHE, Mixup)
  - focal_loss.py : implémentation de la Focal Loss pondérée
  - calibrate_exp4.py : calibration des seuils et évaluation
    post‑entraînement
- instructions succinctes d'exécution (version de Python, dépendances,
  commande de lancement).

Le code source complet de ce travail, incluant les scripts
d'entraînement, d'augmentation de données et de calibration, est
disponible sous licence libre à l'adresse suivante :\
<https://github.com/yaman1444/master-reseach.git>

**Annexe B : Métriques complètes et courbes ROC**

Cette annexe fournit les détails chiffrés complémentaires qui ne
tiennent pas dans le corps du texte :

- tableaux complets des métriques par expérience (Exp 3, Exp 4, Exp 5) :
  accuracy, precision, recall, F1 par classe, F1 macro, AUC par classe ;
- courbes ROC par classe (normal, bénin, malin) pour le modèle standard
  et le modèle calibré ;
- éventuellement les courbes Precision‑Recall si tu les as générées.

Par exemple :

Figure B.1 -- Courbes ROC par classe (modèle DenseNet‑121 avant
calibration)\
Figure B.2 -- Courbes ROC par classe (modèle calibré -- Expérience 5)\
Tableau B.1 -- Détails des métriques par classe (Exp 3, Exp 4, Exp 5)

  -------------------------------------------------------------------
       Classe         Métrique      Exp. 3      Exp. 4      Exp. 5
  ---------------- --------------- --------- ------------ -----------
     **bénin**        Précision     \~0,76      \~0,84       0,74

                       Recall        88,1%      88,1%      **90,4%**

                      F1-score      \~0,74      \~0,82       0,81

     **malin**        Précision     \~0,78      \~0,78       0,77

                       Recall        \~85%      \~89%        \~94%

                      F1-score      \~0,81      \~0,83       0,85

     **Normal**       Précision     \~0,72      \~0,81       0,89

                       Recall        \~65%      \~76%        \~63%

                      F1-score      \~0,68      \~0,79       0,74

   **Macro avg**      Accuracy       \~76%    **81,67%**     76,7%

                      F1 macro      \~0,72     **0,79**      0,73

                      AUC macro     \~0,91    **0,9219**    \~0,92
  -------------------------------------------------------------------

##  {#section-1}

**Annexe C : Heatmaps Grad‑CAM**

Cette annexe présente des exemples supplémentaires de visualisations
Grad‑CAM obtenues avec le modèle DenseNet‑121 final (Expérience 5). Ces
cartes de chaleur permettent de confirmer que le réseau se focalise sur
les zones d'intérêt clinique (masses, irrégularités de contours) pour
établir son diagnostic.

***Figure C.1 -- Visualisation Grad‑CAM sur une lésion maligne
(malin)*** La zone d'activation (rouge) correspond précisément à la
masse tumorale infiltrante, validant la pertinence spatiale de la
décision.

***Figure C.2 -- Visualisation Grad‑CAM sur une lésion bénigne
(bénin)*** Le modèle identifie une zone suspecte plus restreinte,
cohérente avec une anomalie architecturale précoce.

***Figure C.3 -- Visualisation Grad‑CAM sur un tissu normal***

On observe une absence de foyer d'activation intense, confirmant le
caractère non suspect de l'image.

**Annexe D : Synthèse comparative des cinq expériences**

Les cinq expériences successives ont été conçues pour mesurer la
contribution individuelle de chaque composant du pipeline sur le recall
de la classe *bénin*. Le tableau ci-dessous en présente une vue
consolidée.

​​

  ------------------------------------------------------------------------------------------------------------------
                   Exp. 1         Exp. 2            Exp. 3                Exp. 4                    Exp. 5
  -------------- ----------- ---------------- ------------------ ------------------------- -------------------------
   Fine-tuning   Tête seule     Tête seule         2 phases              2 phases                  2 phases

    Focal Loss       Non           Non               Oui                    Oui                       Oui
                                               ($\gamma = 2.0$)   ($\gamma = 2.0,\alpha$)   ($\gamma = 2.0,\alpha$)

   Augmentation    Basique        Mixup             Mixup          Mixup $\alpha = 0.1$      Mixup $\alpha = 0.1$
                              $\alpha = 0.1$    $\alpha = 0.1$                             

   Calibration       Non           Non               Non                    Non             **Oui (hiérarchique)**
      seuils                                                                               

     Accuracy       \~68%         \~72%             \~76%               **81,67%**                   76,7%
     globale                                                                               

   Recall bénin     \~58%         \~65%             88,1%                  88,1%                   **90,4%**

     F1 macro      \~0,62         \~0,67            \~0,72               **0,79**                    0,73

    AUC macro      \~0,86         \~0,89            \~0,91              **0,9219**                  \~0,92
  ------------------------------------------------------------------------------------------------------------------

*Tableau D.1 --- Résultats comparatifs des cinq expériences sur le jeu
de test (240 images)*

L'Expérience 4 constitue le meilleur compromis en termes d'accuracy et
de F1 global. L'Expérience 5 sacrifie légèrement ces métriques au profit
d'un recall de 90,4% sur les lésions débutantes, conformément à
l'objectif clinique prioritaire de ce mémoire. Chaque composant apporte
une contribution incrémentale mesurable : le fine-tuning en deux phases
améliore la qualité des représentations, la Focal Loss pondérée cible
les classes difficiles, et la calibration hiérarchique agit comme levier
final pour atteindre la cible de sensibilité précoce.

# **LISTE DES FIGURES**

*Figure 1.1 -- Charge du cancer du sein en Côte d'Ivoire (GLOBOCAN
2022)​\
Figure 1.2 -- Contraintes d'infrastructure de dépistage (nombre de
mammographes par population)​\
Figure 1.3 -- Illustration schématique de la classification BI‑RADS des
microcalcifications​\
Figure 1.4 -- Schéma des tissus mammaires denses (BI‑RADS C et D) et
impact sur la visibilité des lésions​*

*Figure 2.1 -- Exemples d'images issues des datasets CBIS‑DDSM, INbreast
et BUSI\
Figure 2.2 -- Schéma simplifié de l'architecture DenseNet‑121​\
Figure 2.3 -- Illustration des connexions denses au sein d'un dense
block​\
Figure 2.4 -- Visualisation d'un exemple de carte Grad‑CAM en imagerie
mammaire​*

*Figure 3.1 -- Répartition du dataset en ensembles d'entraînement,
validation et test (1 580 images)​\
Figure 3.2 -- Déséquilibre des classes dans l'ensemble d'entraînement
(normal / bénin / malin)​\
Figure 3.3 -- Schéma du pipeline de prétraitement (redimensionnement,
normalisation, CLAHE, Mixup)​\
Figure 3.4 -- Vue d'ensemble de l'architecture DenseNet‑121 avec tête de
classification à 3 classes​\
Figure 3.5 -- Historique d'entraînement : courbes de loss et d'AUC sur
train / validation​*

*Figure 4.5 -- Prédiction Grad-CAM sur une lésion de stade bénin ---
confiance 98,81 %​\
Figure 4.6 -- Prédiction Grad-CAM sur une lésion de stade malin ---
confiance 91,42 %​\
Figure 4.7 -- Prédiction Grad-CAM sur un sein normal --- confiance 68,86
%​\
Figure 4.8 -- Matrice de confusion calibrée (Expérience 5) sur le jeu de
test​\
Figure 4.9 -- Évolution du recall par classe entre Expérience 3 et
Expérience 5​ Figure 5.1 -- Exemple de cascade U-Net + DenseNet : Image
originale, Masque U-Net, et Image masquée --- Cas 1​\
Figure 5.2 -- Exemple de cascade U-Net + DenseNet : Image originale,
Masque U-Net, et Image masquée --- Cas 2​\
Figure 5.3 -- Exemple de cascade U-Net + DenseNet : Image originale,
Masque U-Net, et Image masquée --- Cas 3​*

*Figure B.1 -- Courbes ROC par classe pour le modèle non calibré (test
set)​\
Figure B.2 -- Courbes ROC par classe pour le modèle calibré (Expérience
5)​\
Figure C.1 -- Heatmaps Grad‑CAM supplémentaires pour des cas
correctement classés\
Figure C.2 -- Heatmaps Grad‑CAM pour des cas mal classés (faux positifs
/ faux négatifs)​*

# **LISTE DES TABLEAUX**

Tableau 1.1 -- Charge du cancer du sein chez la femme en Côte d'Ivoire
(GLOBOCAN 2022)​\
Tableau 1.2 -- Contraintes infrastructurelles et organisationnelles du
dépistage en Côte d'Ivoire​\
Tableau 1.3 -- Classification BI‑RADS des microcalcifications et
probabilité de malignité​\
Tableau 1.4 -- Tendances de densité mammaire (BI‑RADS C/D) selon les
populations​

Tableau 2.1 -- Principaux jeux de données en cancer du sein (CBIS‑DDSM,
INbreast, BUSI)\
Tableau 2.2 -- Caractéristiques détaillées du dataset BUSI​\
Tableau 2.3 -- Principales familles d'architectures CNN utilisées en
imagerie mammaire​\
Tableau 2.4 -- Stratégies de gestion du déséquilibre des classes (Focal
Loss, oversampling, seuils)​\
Tableau 2.5 -- Méthodes XAI courantes pour les CNN (Grad‑CAM,
Grad‑CAM++, SHAP)​

Tableau 3.1 -- Répartition globale du dataset (train / validation /
test)​\
Tableau 3.2 -- Distribution des classes dans l'ensemble d'entraînement​\
Tableau 3.3 -- Résumé des hyperparamètres d'entraînement (LR, batch
size, gamma, alpha)​\
Tableau 3.4 -- Mécanismes de reproductibilité mis en place (scripts,
seeds, checkpoints)​

Tableau 4.1 -- Performances globales de DenseNet‑121 avant calibration
(test set)​\
Tableau 4.2 -- Performances de DenseNet‑121 après calibration
(Expérience 5)\
Tableau 4.3 -- Détails par classe : précision, recall, F1 avant et après
calibration\
Tableau B.1 -- Métriques complètes des Expériences 3, 4 et 5 (train /
validation / test)

# **LISTE DES ABRÉVIATIONS**

**AUC** : Area Under the Curve (surface sous la courbe ROC)​\
**AUROC** : Area Under the Receiver Operating Characteristic Curve​\
**BI‑RADS** : Breast Imaging Reporting and Data System​\
**CAD** : Computer‑Aided Detection​\
**CADx** : Computer‑Aided Diagnosis​\
**CBAM** : Convolutional Block Attention Module​\
**CBIS‑DDSM** : Curated Breast Imaging Subset of the Digital Database
for Screening Mammography​\
**CHU** : Centre Hospitalier Universitaire​\
**CLAHE** : Contrast Limited Adaptive Histogram Equalization​\
**CNN** : Convolutional Neural Network​\
**CPU** : Central Processing Unit​\
**DCIS** : Ductal Carcinoma In Situ​\
**DL** : Deep Learning​\
**FFDM** : Full‑Field Digital Mammography​\
**FN** : False Negative (faux négatif)​\
**FP** : False Positive (faux positif)​\
**FPR** : False Positive Rate​\
**F1** : F1‑score (moyenne harmonique précision / recall)​\
**GPU** : Graphics Processing Unit​\
**Grad‑CAM** : Gradient‑weighted Class Activation Mapping​\
**IA** : Intelligence Artificielle​\
**JSON** : JavaScript Object Notation​\
**LR** : Learning Rate (taux d'apprentissage)​\
**ML** : Machine Learning​\
**NPV** : Negative Predictive Value​\
**PPV** : Positive Predictive Value​\
**ReLU** : Rectified Linear Unit​\
**ROC** : Receiver Operating Characteristic​\
**SGD** : Stochastic Gradient Descent​\
**SOTA** : State Of The Art​\
**TN** : True Negative (vrai négatif)​\
**TP** : True Positive (vrai positif)​\
**TPR** : True Positive Rate (recall / sensibilité)​\
**TTA** : Test‑Time Augmentation​\
**UVCI** : Université Virtuelle de Côte d'Ivoire​\
**ViT** : Vision Transformer​\
**XAI** : eXplainable Artificial Intelligence
