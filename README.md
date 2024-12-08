# building-perceptron
## The project


Ce projet vise à présenter les bases du deep learning, en implémentant un algorithme du perceptron. Pour cela, il est nécessaire de comprendre les notions de Machine Learning et de Deepl Learning, toutes deux liées à l'Intelligence Artificielle, et de pouvoir clairement les distinguer.

### Définitions

Le Machine Learning (apprentissage automatique), est branche de l'IA qui permet, à travers des algorithmes et des modèles statistiques, d'analyser des données (ensemble d'apprentissage) pour en apprendre de ces données, afin d'appliquer le modèle d'apprentissage pour prendre des décisions sur d'autres données non présente dans l'ensemble d'apprentissage. 

Le Deep Learning quant à lui, même s'il est une forme de Machine Learning, consiste à structurer des algorithmes en couches, pour créer un réseau de neurones artificiel, capables d'apprendre et de prendre des décisions intelligentes de façon autonome. On parle de réseau de neurones artificiel, car il s'inspire du fonctionnement du cerveau humain.


### Comparaison

A partir de leurs définitions respectives, on peut ressortir des différences particulières entre ces deux notions. En termes de complexité, le Machine Learning fait souvent appel à des modèles simples (regressions, KNN, arbres de décisions, SVM, etc.). Quant au Deepl Learning, il nécessite l'utilisation de modèles beaucoup plus complexes, due à la profondeur des réseaux de neurones (CNN, GAN, etc.)
De plus, les algorithmes de Deep Learning sont capables d'une sélection de variables (features) presque totalement automatique, à travers la mesure de l'importance de chaque caractéristique. La différence avec le Machine Learning est que dans ce cas, cette sélection est souvent basée sur un choix de l'utilisateur. Même si ce choix se fait à partir de métodes statistiques, il reste en grande partie subjectif face à la sélection faite en Deep Learning.
Par conséquent, les algorithmes de Deep Learning ont besoin de plus gros volumes de données pour une sélection de variables plus pertinente (problème de dimensionalité); ce qui représente aussi une grande différence avec le Machine Learning.

### Situations où il faut privilégier l'un ou l'autre

On choisira des méthodes de Machine Learning plûtot que celles de Deep Learning lorsqu'on ne dispose pas d'assez de données, ce qui peut être difficile à déterminer. En général, l'analyse de données complexes (en taille et/ou en structure de données) oriente le travail vers le Deepl Learning, surtout si on dispose d'une puissance de calcul considérable.
De plus, les modèles basiques de Machine Learning (régressions par exemple), permettent d'interprèter clairement les résultats en termes d'effet causal des variables explicatives sur la variable cible; ce qui est n'est pas le cas lorsqu'on utilise des modèles de Deep Learning. Pour cela, on utilisera des modèles de Machine Learning si on cherche à interprèter le modèle d'une façon causale, et les modèles de Deep Learning lorsque l'objectif principal est la sélection de caractéristiques pertinentes.


Une fois que les notions de Machine Learning et de Deep Learning ont été éclaircies, nous présentons trois (03) différentes applications du Deep Learning:

# Application 1: Traitement du Langage Naturel (Natural Language Processing)

Le Deep Learning est utilisé dans les tâches de traduction automatique de texte, à travers l'algorithme RNN (réseaux de neurones récurrents). Le mécanisme se base sur la capacité de l'algorithme à garder en mémoire la séquence de texte précédente, et à se baser sur la traduction de cette séquence pour traiter la séquence suivante. Par exemple, pour traduire une phrase de 10 mots, l'algorithme va en premier parcourir la phrase, et former un vecteur de taille 10 comportant les mots de la phrase. Ensuite, ce vecteur sera utilisé pour traduire chacun des mots dans la langue cible, en considérant les traductions déjà générées des séquences précédentes, ainsi que la probabilité que le mot suivant ait une certaine signification selon le contexte de la phrase.


# Application 2: Reconnaissance d'objets (avec Google Lens par exemple)

Les réseaux de neurones convolutionnels (CNN), dans leur forme améliorée R-CNN (regions with CNN features), sont utilisées dans la détection d'objets. Pour une image en entrée, à l'entraînement, la première étape est d'utiliser un algorithme pour générer environs 2000 régions ayant une forte probabilité de contenir les objets recherchés. Ensuite, dans une étape de classification, les régions découpées sont redimensionnées, et le CNN est utilisé pour extraire les features (caractéristiques des objets recherchés) de l'image initiale entière. On utilise ensuite un SVM pour classer chaque région dans l'une des catégories d'objets prédéfinies (un SVM par catégorie), et un BBR (Bounding Box Regressor) pour prédire et affiner les coordonnées de chaque régions selon les caractéristiques sélectionnées.

# Application 3: NotebookLM de Google

Cet outil d'IA de Google permet de faire des recherches dans des documents en détectant le texte dans des images. Après un prétraitement des images visant à supprimer le bruit et à améliorer le texte, un réseau de neurones convolutionnel (CNN) est utilisé pour extraire le texte des images. Ce texte est ensuite analysé et indexé pour permettre de retrouver plus rapidement les réponses aux questions posées par l'utilisateur.

En s'intéressant particulièrement au perceptron dans le cadre de ce projet, voici quelques éléments permettant de comprendre son fonctionnement.

# Définition du perceptron et lien avec le neurone biologique

Le perceptron est un modèle mathématique inspiré par le fonctionnement du neurone biologique, introduit par Frank Rosenblatt en 1957, et considéré comme le précurseur des réseaux de neurones profonds. Il fonctionne comme un neurone biologique, car il reçoit une information en entrée (valeurs numériques), et produit une sortie unique; comme le neurone biologique qui reçoit des signaux par les dendrites, et emet un signal par l'axone. De plus, l'information en entrée a un poids dans les deux cas: pour le neurone biologique, il s'agit de l'intensité de la connexion entre les synapses de deux ou plusieurs neurones connectés; le perceptron quant à lui, assigne des poids à caque entrée, représentant l'importance de cette entrée dans la réponse en sortie. Enfin, un neurone biologique ne déclenche un signal que si le potentiel électrique accumulé dépasse un certain seuil, ce qui est très semblable au perceptron qui, à travers une fonction seuil (fonction d'activation), détermine la sortie binaire.

# Fonction matématique du perceptron

La fonction mathématique du perceptron est: y = f(∑(wi * xi) + b, avec:
- y la sortie du perceptron (0 ou 1)
- f() la fonction d'activation
- wi le poids associé à la i-ème entrée
- xi la valeur de la i-ème entrée
- b le biais
 Le biais est à différencier du seuil d'activation, même s'ils représentent tous deux une valeur constante ajoutée à la somme pondérée des entrées. En effet, le biais est un paramètre du modèle qui peut être appris pendant l'entraînement, alors que le seuil d'activation est prédéterminé. 

 En ce qui concerne l'usage du perceptron, il permet de classer des données en deux catégories; par exemple, il peut être utilisé pour la détection de spams, la reconnaissance d'images, et le traitement du langage naturel (analyse de sentiment positif ou négatif dans un discours).

 # Règles d'apprentissage du perceptron

 - Apprentissage par Renforcement Hebbien (Perceptron à seuil): Pour une entrée Ak = (xk, ŷk), mise à jour des poids et du seuil en fonction de la différence entre la sortie calculée y et la sortie désirée ŷ; 
wi_nouveau = wi_ancien + λ * (ŷk - yk) * xik
𝜗_nouveau = 𝜗_ancien + λ * (ŷk - yk) avec:
λ le taux d'apprentissage, un paramètre positif qui contrôle l'ampleur de la correction;
xik est la valeur de la i-ème composante du vecteur d'entrée xk.
  Règle de décision:
    - si la sortie désirée est 1 et la sortie calculée est 0, les poids associés aux entrées positives sont augmentés et le seuil est diminué;
    - si la sortie désirée est 0 et la sortie calculée est 1, les poids associés aux entrées positives sont diminués et le seuil est augmenté.

- Apprentissage par Descente du Gradient (Perceptron Continu)
Pour les perceptrons continus utilisant une fonction d'activation différentiable, l'apprentissage se fait en minimisant une fonction objectif (E) qui mesure l'erreur entre la sortie du perceptron et la sortie désirée pour tous les exemples de l'ensemble d'apprentissage. La méthode de descente du gradient est utilisée pour trouver les valeurs des poids et du seuil qui minimisent cette fonction.
Fonction Objectif : E(w, 𝜗) = (1/2) * ∑(yk - ŷk)^2
où la somme s'effectue sur tous les exemples de l'ensemble d'apprentissage.

Formules de mise à jour :
wi_t+1 = wi_t - λ * ∂E/∂wi
𝜗_t+1 = 𝜗_t - λ * ∂E/∂𝜗
où :
t représente l'itération actuelle.
∂E/∂wi et ∂E/∂𝜗 représentent les dérivées partielles de la fonction objectif par rapport aux poids et au seuil, respectivement.
La méthode ajuste les poids et le seuil dans la direction opposée au gradient de la fonction objectif. Cela permet de converger vers un minimum de la fonction objectif, ce qui correspond à une minimisation de l'erreur de classification.


# Fonction d'activation classique

Il s'agit de la fonction sigmoïde, définit comme suit: f(x) = 1 / (1 + e^(-x))

où x est la somme des produits des poids des connexions entre les entrées et la sortie, et e est la base naturelle (≈ 2.71828).

Elle prend les valeurs 0 ou 1 selon les entrées du perceptron.

# Processus d'entraînement du perceptron

Il se déroule en sept (07) étapes:
a- initialisation des poids et du seuil avec des valeurs aléatoires ou prédéfinies;

b- introduction d'un exemple d'apprentissage composé d'un vecteur d'entrée (xk) et de la sortie désirée (ŷk);

c- calcul de la sortie potentielle (ξk) en effectuant la somme pondérée des entrées et en y ajoutant le seuil d'activation:
ξk = ∑ (wi * xik) + 𝜗; la sortie du perceptron (yk) est ensuite déterminée en appliquant une fonction d'activation au potentiel;

d- comparaison de la sortie du perceptron yk et de celle de départ ŷk;

e- mise à jour des poids et du seuil si la sortie calculée est différente de la sortie désirée, dans le but de rapprocher la sortie calculée de la sortie attendue;

f- répétition des étapes b à e : les étapes d'introduction d'un exemple, de calcul de la sortie, de comparaison et de mise à jour des poids sont répétées pour tous les exemples de l'ensemble d'apprentissage;

g- critère d'arrêt: Le processus d'apprentissage se poursuit jusqu'à ce qu'un critère d'arrêt soit atteint; cela peut être:
  - tous les exemples correctement classifiés: le perceptron a appris à classifier correctement tous les exemples de l'ensemble d'apprentissage;
  - nombre maximal d'itérations atteint: le processus d'apprentissage s'arrête après un nombre prédéfini d'itérations, même si tous les exemples ne sont pas correctement classifiés;
  - gradient de la fonction objectif inférieur à une valeur seuil: pour les perceptrons continus, l'apprentissage s'arrête lorsque le gradient de la fonction objectif est suffisamment petit, indiquant que l'erreur de classification est minimisée.

# Limites du perceptron

a- Séparabilité linéaire: le perceptron ne peut classifier correctement que des ensembles de données linéairement séparables. Cela signifie qu'il est incapable de trouver une solution pour des problèmes où les classes ne peuvent pas être divisées par une ligne droite (en deux dimensions) ou un hyperplan (en dimensions supérieures);

b- Classification binaire: le perceptron ne peut être utilisé que pour des tâches de classification binaire; au-delà de deux classes, il faut un réseau avec plusieurs couches de neurones;

c- Données indépendantes et identiquement distribuées (i.i.d): l'utilisation du perceptron suppose que les données sont indépendantes et identiquement distribuées, ce qui signifie que chaque point de données est indépendant des autres et provient de la même distribution de probabilité; cette hypothèse peut ne pas être valable dans de nombreux scénarios réels, où les données peuvent présenter des dépendances et provenir de distributions différentes.

d- Difficulté à gérer les données avec des bruits ou avec desvaleurs aberrantes: le perceptron est sensible aux données bruitées, c'est-à-dire aux données contenant des erreurs ou des incohérences; les exemples erronés peuvent perturber l'apprentissage et empêcher le perceptron de trouver une solution optimale.

e- Sensibilité aux variations de poids et au choix de la fonction d'activation: utiliser par exemple des poids aléatoires

f- Surapprentissage: le perceptron continue d'apprendre même lorsque les données sont traitées et enregistrées;  cela peut engendrer des difficultés liées à une suradaptation et à une surmodélisation.



### Project Members
Yann Sasse
Amina Sadio
Julien Ract-Mugnerot

### Requirements

python pandas etc


## Note à l'équipe:

Méthode pour importer un fichier sur une autre branche:
aller sur la branche vers laquelle l'on veut copier le fichier
git checkout NomBrancheOuEstLeFichier -- file(avec extensions)



- Idéalement on ne va travailler que sur deux branches principales et le moins de fichiers possibles, évitons les branches julien et julien_notebook.ipynb
  - Class_Perceptron
  - Data_Exploration --> Data_Preprocessing --> Feature_Selection ...
`Ce serait bien qu'on change de branche à la complétion de chaque étape sur la partie notebook.`


- Réaliser les commentaires associés au notebook en même temps que chaque étape produite
- Si les push à chaque fonctionnalité peuvent suivrent la forme "[Create|Update|Delete] Data Import" recommandé par Kawther l'année dernière, ce serait top.
- Personellement j'aimerais beaucoup qu'on utilise activement le kanban github pour savoir ou on en est
- Réaliser la classe à partir de ce tuto pour démarrer: https://www.youtube.com/watch?v=-KLnurhX-Pg
- absolument utiliser Boruta pour la feature selection, si on a le temps chercher d'autres technique
- Réalisation de multiples tests unitaires pour la classe python
