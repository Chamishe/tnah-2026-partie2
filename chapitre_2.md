
#  🙤 **PARTIE 2** 🙧

  

#  Extraction structurée des indices géographiques dans les métadonnées des photographies d'[Eugène Atget](https://fr.wikipedia.org/wiki/Eug%C3%A8ne_Atget)

  
  

🙑 Au-delà de son volume, l'œuvre d'Eugène Atget frappe par la méticulosité de son classement. Véritable archiviste de son propre travail, l’auteur a systématiquement indexé ses clichés, les structurant en séries et en albums thématiques. Chaque cliché est doté d'un titre décrivant son sujet et sa localisation, souvent très précisément.

  

Ces titres, systématiquement relevés et transcrits, font parti des métadonnées associées à chaque photographie. Les archivistes ne se sont cependant pas arrêté là et ont enrichi la description structurée de chaque cliché avec des thèmes issus du thésaurus Rameau.

  

La richesse géographique de ces métadonnées permet aujourd'hui d'envisager une cartographie du fonds dans l’espace parisien. Cette approche, image par image, offrirait un parcours inédit au cœur de la capitale et renouvellerait notre regard sur l’œuvre du photographe.

  

🙑 **Rappel**. Dans la [partie 1](https://github.com/HueyNemud/tnah-2026-partie1), nous avons exploré le graphe de connaissances de la bibliothèque nationale de France, publié sur data.bnf.fr, analysé le schéma de métadonnées *WEMI* utilisé pour structurer les métadonnées descriptives des œuvres, et extrait pour chaque photographie un graphe RDF contenant les métadonnées la décrivant. Dans le chapitre 4, les graphes individuels des photographies ont été enregistrés au format Turtle sur le disque, dans un dossier nommé `photographies/`

##  🙤 Objectifs

Cette seconde partie guide la mise en place d'un processus d'extraction de l'information géographique contenue dans les métadonnées des photographies d'Eugène Atget récoltées dans la première partie.

Elle se décompose en deux étapes :

1. D'abord enrichir les graphes de métadonnées des photographies avec les **thèmes Rameau** qui leurs sont associés, afin d'obtenir le plus possible d'indications géographiques. Cette première étape est l'occasion d'un **exercice de lecture de code** Python.
  
2. Ensuite, **extraire l'information géographique pertinente** dans le titre et des thèmes de chaque photographie pour obtenir un ensemble d'indications de localisation qui pourront être utilisé pour placer la photographie dans Paris et ses alentours. Cette tâche de **traitement automatique du langage naturel** sera réalisé avec un **grand modèle de langage génératif**.

Ce chapitre 2 porte sur la **seconde étape**.

Légende des pictogrammes utilisés :

| Picto. | Légende |
|  ------  |  -----------------------------------------  |
| 🎬 | Action à réaliser : à vous de jouer ! |
| 💡 | Suggestion d'action complémentaire |
| ⚠️ | Avertissement |
| ℹ️ | Information supplémentaire ou astuce |
| 📚 | Ressources : documentation, article, etc. |

<hr/>

##  🙤 Chapitre 2 : extraction structurée de l'information géographique contenue dans les métadonnées des photographies

**⚠️ Prérequis**
- Avoir terminé le chapitre 1 de la partie 2 et exécuté le script Python `fetch_themes.py` pour enrichir tous les graphes de photographies dans le dossier `photographies/` avec les labels de leurs thèmes Rameau.
- le dossier `photographies_avec_themes/` doit exister et doit contenir les **sérialisations Turtle des graphes de chaque photographie enrichies** de leurs thèmes Rameau.

###  Motivation
Notre but est de réussir à placer chaque photographie d'un lieu sur une carte, à l'endroit où se trouvait ce lieu.
Pour cela il faut **géocoder** la photographie, c'est à dire lui lui associer des **coordonnées géographiques**, en l’occurrence celles du lieu pris en photo par Atget.

On a besoin pour cela de glaner **l'information géographique** qui se trouve dans les métadonnées d'une photographie, ce qui - on l'espère - fournira assez d'indices géographiques sur l'emplacement de cette photographie.

Dans le chapitre 1, on a vu que le **titre** de la photographie et ses **thèmes Rameau** contenaient des indices géographiques.

Parce que les grands modèles de langages sont conçus pour exploiter -entre autre- du texte, nous n'allons pas travailler directement avec les graphes de connaissance des photographies, mais avec une forme "sérialisée" en texte : leur rapport d'enrichissement produit par la fonction `build_summary_report()` dans le script `fetch_themes.py`.

Voici par exemple le rapport sur l'enrichissement de la photographie du [Cabaret du Soleil d'or](https://gallica.bnf.fr/ark:/12148/btv1b10506998t#)  :
```raw
=== PHOTO : Au Soleil d'or : 84 [quatre-vingt-quatre] Rue S.t Sauveur (Modifié), [photographie] ===
Lien : http://data.bnf.fr/ark:/12148/cb40268281c#about
Thèmes assignés:
• « Dans l'art » - altLabels : « Représentation dans l'art », « Dans la sculpture », « Dans la peinture », « Représentation iconographique », « Dans les arts graphiques »
• « Cafés » - altLabels : « Cafés-bars », « Débits de boissons », « Estaminets », « Brasseries (cafés) », « Zincs (cafés) », « Bistrots », « Cafés publics », « Cafés (établissements) », « Bars »
• « Paris (France) »
• « Paris (France) -- Rue Saint-Sauveur » - altLabels : « Rue Saint-Sauveur (Paris, France) », « Saint-Sauveur, Rue (Paris, France) »
• « Enseignes » - altLabels : « Signes et indications », « Enseignes commerciales »
• « Ferronnerie d'art » - altLabels : « Serrurerie d'art », « Fer forgé, Objets en », « Fer ornemental », « Ferronnerie architecturale », « Ferrures », « Ferronneries », « Ferronnerie décorative », « Fer forgé », « Objets en fer forgé », « Ferronnerie (architecture) »
• « Soleil » - altLabels : « Physique solaire »
```
  
Une simple lecture de ce rapport permet de repérer un certain nombre d'indices géographiques, qu'on peut intuitivement organiser par granularité pour former une **hiérarchie spatiale** qui décrit à plusieurs échelles l'emplacement du lieu dans Paris :
```raw
📍 France [pays]
	└── 🏙️ Paris [ville]
		└── 🛣️ rue Saint-Sauveur [rue]
			└── 🏠 n° 84 rue Saint-Sauveur [adresse]
				└── ☀️ « Au Soleil d'or » [enseigne / toponyme]
```

Cette opération en apparence simple enchaîne plusieurs taches qui mobilisent des capacités cognitives humaines typiquement difficiles à transférer sous la forme de programme informatique : 
- **identifier** l'information géographique dans le titre et les thèmes nécessite de comprendre la langue française, de connaitre la géographie parisienne, ...
- **extraire** cette information géographique nécessite de réussir à la séparer de son contexte et au besoin à la réinterpréter, par exemple pour que "84 [quatre-vingt-quatre] Rue S.t Sauveur"  soit compris comme l'adresse du n°84 rue Saint-Sauveur".
- **organiser** cette information de manière hiérarchique implique également une connaissance implicite importante: comment fonctionne une adresse, comment les humains structurent hiérarchiquement les lieux, etc.

Réaliser cet enchaînement de tâches avec un ordinateur est un cas typique de **traitement automatique du langage naturel**.
On pourrait imaginer la programmer naïvement sous la forme d'une chaîne de traitement, en utilisant par exemple des techniques à base de règles, mais cela serait à la fois très fastidieux et très fragile aux variations dans les formes de description des métadonnées.

Aujourd’hui, utiliser un **grand modèle de langage (LLMs) génératifs** pour réaliser cet enchaînement de taches en **une seule étape** est généralement l'approche la plus efficace. 

Nous allons donc utiliser un LLM pour **extraire et organiser hiérarchiquement l'information géographique** contenue dans le titre et les thèmes Rameau des photographes.

Cette expérimentation sera réalisée avec les grands modèles de langages de **[Mistral AI](https://mistral.ai/fr)**.

###  Premier essai naïf
En tant qu'utilisateur, le fonctionnement d'un modèle de langage est simple : c'est modèle statistique qui prend en entrée une information - par exemple un texte -, et génère un nouveau texte qui est conditionné par le contenu informationnel de l’entrée. 

Ces modèles sont extrêmement performants pour réaliser de nombreuses taches de traitement automatique du langage naturel, car leur capacité d'attention et les connaissance stockées dans leur mémoire (apprise) les rendent capables de traitements complexes nécessitant des connaissances implicites importantes.

Ces capacités d'analyse sont maintenant bien connues et les LLM sont explicitement entraînés pour être utilisés comme outils de traitement que l'on peut guider à l'aide d'**instructions** (le *prompt*).

Testons les capacités du LLM de Mistral pour extraire la hiérarchie géographique du lieu décrit dans les métadonnées d'une photographie.
Commençons avec le mode d'accès le plus grand public au LLM principal de Mistral :  le chatbot _Le Chat_.

Essayons de faire extraire au LLM la hiérarchie géographique parsemée dans le rapport sur la photographie du [abaret du Soleil d'or](https://gallica.bnf.fr/ark:/12148/btv1b10506998t#) donné dans la section **Motivation**.

Pour guider un LLM, il faut un *prompt* décrivant la tache qu'il doit réaliser.
S'il n'existe pas de "bible" de la rédaction de prompt, tous les LLMs sont entraînés pour comprendre des *prompts* dont structure générale est la suivante :

> a. Assigner un rôle au modèle pour la tache.
> b. Décrire la tâche à réaliser
> c. Si nécessaire, donner des règles spécifiques pour gérer les cas complexes, ambigus, etc.

Voici un premier *prompt* très simple qu'on peut utiliser pour extraire la hiérarchie géographique à partir d'un rapport d'enrichissement de photographie :

```raw
# Rôle
Tu es un expert en extraction d'information géographique dans des métadonnées patrimoniales.

# Tâche
À partir d'un résumé descriptif d'une photographie ancienne, tu dois :
1. identifier les entités géographiques qui renseignent sur **la localisation du sujet de la photographie** dans l'espace ;
2. lister ces entités les uns après les autres.

# Règles
- les entités doivent être triée de la plus précise à la plus générale.

Le résumé à traiter sera donné dans le prochain input.
```

> ℹ️ Notez qu'on ne donne ici **aucun exemple** de traitement. Cette stratégie fondée uniquement sur des règles s'appelle **0-shot prompting**.

> 🎬 Rendez vous sur https://chat.mistral.ai/chat et donnez ce *prompt* au *chatbot*. Donnez ensuite le rapport d'enrichissement sur le cabaret du Soleil d'Or.

Vous devriez constater que même avec un *prompt* grossier, le modèle fournit une réponse déjà très satisfaisante contenant une partie de la hiérarchie souhaitée.
Par exemple :
```raw
Pour cette photographie, les entités géographiques permettant de localiser
 le sujet sont les suivantes :
1.  84 Rue Saint-Sauveur**
2.  Rue Saint-Sauveur (Paris, France)**
3.  Paris (France)
Si vous avez besoin d'une autre analyse, n'hésitez pas à me le demander !
```
  > ℹ️ Du fait de la nature stochastique d'un LLM, sa réponse peut varier.

Toutefois, le *prompt* contraint très peu la réponse du LLM : pas de format imposé, pas de liste de niveau hiérarchiques à extraire.
Laissé libre, le modèle a "décidé" d'organiser la hiérarchie en liste numérotée, d'ajouter des commentaires, ignorer le toponyme ou ajouter des compléments entre parenthèses.

Cela ne pose pas de problèmes de compréhension à un humain, mais n'oublions pas que le notre but est d'extraire une hiérarchie exploitable dans un processus **automatique** de géocodage.
Il faut donc que la sortie du LLM ne soit pas du simple texte, mais un texte formaté, standardisé, compréhensible par un programme.
C'est ce qu'on appelle de  **l'extraction structurée** d'information.

###  Extraction structurée avec _Le Chat_
Pour produire une réponse dans un format précis, interprétable informatiquement et qui contienne tous les niveaux hiérarchiques souhaités, nous devons guider bien plus strictement le modèle.

Une manière simple et particulièrement efficace consiste à ajouter au *prompt* au moins un exemple de rapport et la sortie attendue.
Cette stratégie de guidage contextuel par l'exemple se nomme *few-shot prompting*.
  > ℹ️ On trouve aussi parfois le terme de *few-shot training*, mais cette appellation porte à confusion et tend à disparaître. Si les réseaux de neurones profonds sont bien entraînés avec des exemples, cela n'a rien a voir avec le *prompting*. Les exemples dans le *prompt* font seulement partie du contexte accessible au modèle durant la génération du nouveau texte - cela guide son attention courante et son comportement, mais il n'apprend rien et ne stocke aucune nouvelle connaissance dans sa mémoire.
  
A priori, tout format structuré est envisageable pour la hiérarchie inférée.
Cependant, les LLMs sont généralement plus performants pour produire du JSON, parce qu'ils ont été entraînés pour les taches d'extraction structurée spécifiquement avec ce format.

Voici une représentation JSON possible de la hiérarchie donnée en **motivation** : 
```json
{
	"toponyme": "Au Soleil d'Or",
	"adresse": "84 rue Saint-Sauveur",
	"voie" : "rue Saint-Sauveur",
	"ville": "Paris",
	"pays": "France"
}
```

> 🎬 Modifiez le *prompt* d'extraction pour : 
> 1. Insérer une nouvelle règle : "- la réponse doit contenir **uniquement** du JSON suivant le schéma donné en exemple."
> 2. Ajouter une quatrième section nommée  `# Exemple` formatée ainsi 
```raw
# Exemple
**Résumé descriptif **
<Ajoutez le rapport de la section **motivation** sur le cabaret du Soleil d'Or>

**Réponse JSON**
<Ajoutez ici la représentation JSON donnée ci-dessus>
```
> Créez un **Nouveau Chat** pour ne pas biaiser Mistral avec vos précédents messages, puis donnez le nouveau *prompt*.

Pour tester les performances de ce nouveau *prompt* nous devons utiliser un nouveau rapport d'enrichissement puisque celui du cabaret du Soleil d'Or est déjà donné comme exemple.

> 🎬 Testez avec le rapport du "Bon Puits" :
```raw
=== PHOTO : Au Bon Puits : Rue Michel Le Comte 36 (Disparu en 1904), [photographie] ===
Lien : http://data.bnf.fr/ark:/12148/cb40268303v#about
Thèmes assignés:
• « Vin -- Industrie et commerce » - altLabels : « Commerce vinicole », « Industrie viticole », « Commerce viticole », « Production viticole », « Production vinicole », « Industrie vinicole »
• « Paris (France) -- Rue Michel-le-Comte » - altLabels : « Rue Michel-le-Comte (Paris, France) », « Michel-le-Comte, Rue (Paris, France) »
• « Paris (France) »
• « Enseignes » - altLabels : « Signes et indications », « Enseignes commerciales »
• « Ferronnerie d'art » - altLabels : « Ferronnerie architecturale », « Ferrures », « Serrurerie d'art », « Ferronneries », « Fer forgé, Objets en », « Fer ornemental », « Ferronnerie décorative », « Ferronnerie (architecture) », « Fer forgé », « Objets en fer forgé »
```
> Le modèle doit répondre la hiérarchie JSON suivante : 
```json
{
	"toponyme": "Au Bon Puits",
	"adresse": "36 rue Michel-Le-Comte",
	"voie": "rue Michel-Le-Comte",
	"ville": "Paris",
	"pays": "France"
}
```
> C'est mieux, non ? 🙂 On obtient le format attendu, avec tous les niveaux hiérarchiques souhaités, triés dans le bon ordre.

Voyons maintenant comment automatiser ce traitement en Python grâce à la bibliothèque `mistralai` publiée par Mistral.

###  Extraction structurée par LLM en Python avec `mistralai`
Commençons par installer la bibliothèque avec `uv`.
> 🎬 Dans un terminal, placez vous dans le dossier de travail `tnah-2026-partie2` et exécutez
```bash
uv add mistralai
```
Créons ensuite un fichier de script Python qui contiendra le processus complet d'extraction structurée.

> 🎬 Dans le même dossier, créez un nouveau fichier Python nommé `structured_extraction.py`. Par exemple depuis le terminal :
```bash
touch structured_extraction.py
```

La [documentation générale de Mistral](https://docs.mistral.ai) présente et illustre par l'exemple comment utiliser la bibliothèque `mistralai` pour faire de l'extraction structurée.

> 🎬 Rendez-vous dans la documentation de Mistral, sur https://docs.mistral.ai, et cherchez dans le menu gauche l'entrée *Structured Outputs* puis *JSON Mode*.
> Lisez la ligne de présentation. Comprenez-vous quelle est la spécificité de ce *JSON Mode* ?

La page de documentation donne un exemple complet d'interaction avec le modèle LLM `
mistral-large-latest`  hébergé sur leurs serveurs.

> 📚 `mistral` est la famille de modèle, `mistral-large` est le plus modèle ayant le plus de paramètres, généralement le plus puissant. Le suffixe  `-latest` désigne la version la plus récente disponible.

> 🎬 Copiez cet exemple dans le fichier `structured_extraction.py` et :
```python
# 1. Commentez **momentanément la ligne suivante 
#api_key = os.environ["MISTRAL_API_KEY"]
# Remplacez la par :
api_key = VOTRE_CLÉ_MISTRAL_ICI 

... # le reste de l'exemple est inchangé

# 2. Ajoutez à la fin du script la ligne suivante pour afficher le résultat 
# de la requête envoyée à Mistral :
print(chat_response.choices[0].message.content)
```
> 🎬 Exécutez le script et vérifiez que le résultat est le même que sur la page de documentation.
```bash
uv run structured_extraction.py
```
Nous voilà avec un script minimaliste mais fonctionnel pour utiliser le LLM `mistral-large` de manière programmatique.


> ℹ️ Notez que c'est la présence du paramètre `response_format = {"type": "json_object"}` dans l'appel à `client.chat.complete()` qui **contraint** le modèle à produire un résultat JSON. Sans lui le modèle se comporterait exactement comme le chatbot "Le Chat", c'est à dire sans aucune garantie stricte de produire un JSON correct.

###  Extraction d'une hiérarchie géographique avec Mistral
Adaptons maintenant le script pour notre tache d'extraction de hiérarchie géographique !👏

Dans l'exemple, après avoir créé un client `Mistral` représentant la connexion au LLM distant, on définit la variable `messages` qui est une liste de messages à envoyer au modèle.
Chaque message est représenté par un dictionnaire contenant deux clés : `"content"` et `"role"` . 
La clé *content* est triviale : c'est le contenu du message qui est donné au modèle.
La clé *role* peut prendre plusieurs valeurs; deux nous intéressent ici :
- `"role": "user"` : le LLM va considérer que le message est celui d'un utilisateur, et y porter une attention passagère. C'est typiquement le rôle adéquat pour envoyer le **rapport d'enrichissement** à traiter.
- `"role": "system"` : permet de définir un **system prompt**, c'est à dire une instruction générale que le modèle va conserver à "l'esprit" toute la durée de l'échange. Ce rôle est spécialement adapté pour donner les **instructions de traitement** au modèle.

> 🎬 Modifiez les messages stockés dans la variable `messages` pour :
> 1. Donner dans un premier message avec le rôle **system** le prompt *few-shot* créé dans la section **Extraction structurée avec _Le Chat_**
> 2. Donner dans un second message avec le rôle **user** le rapport du "Bon Puits".
> 
> Exécutez à nouveau le script et vérifiez qu'il affiche bien la hiérarchie JSON dans le terminal !
 
###  Automatisation de l'extraction structurée
Jusqu'ici, nous avons assigné manuellement le rapport d'enrichissement à traiter.
Allons un cran plus loin en utilisant les fonctions définies dans `fetch_themes.py`, vues dans le chapitre 1, pour créer dynamiquement un rapport et l'envoyer à Mistral.

Nous avions utilisé `fetch_themes.py` comme un script exécutable, mais nous pouvons également l'utiliser comme un **module python** dont on peut importer les fonctions.

> 🎬 Importez dans `structured_extraction.py` les fonctions `import_turtle_file()` `build_summary_report()` du fichier `fetch_themes.py` :
```python
from fetch_themes import build_summary_report, fetch_themes
```

Nous pouvons ensuite utiliser ces fonctions pour lire un graphe de photographie et créer son rapport d'enrichissement.

> 🎬 Après la création du client Mistral, utilisez les deux fonctions importées pour lire un fichier de graphe enrichi de votre choix depuis le dossier `photographies_avec_themes/` et créer son rapport d'enrichissement et : 
> 1. Stockez ce rapport dans un variable nommée `report`
> 2. Affichez la avec `print()`.
> 3. Remplacez dans la déclaration des messages le contenu du message de rôle *user* par la variable `report`
> 4. Exécutez le script pour vérifier qu'il traite bien le fichier de graphe que vous avez choisi !

###  Traitement en masse des graphes de photographie
Reste une ultime étape : **traiter tous les graphes** et **enregistrer le résultat JSON sur le disque** pour la phase suivante de géocodage.

Il manque pour cela deux éléments :
1. une boucle pour traiter chaque fichier de graphe du dossier  `photographies_avec_themes/` ;
2. une fonction d'enregistrement de la réponse du modèle en JSON.

Commençons par le premier élément, où nous pouvons reprendre exactement la même logique que dans le script `fetch_themes.py`. N'hésitez pas à "piocher" dans ce script pour vous aider.

> 🎬 Ajoutez l'import de la classe `Path` de `pathlib`:
```python
from pathlib import Path
```
> 🎬 Créez ensuite une variable `DIR` qui contient le chemin vers le dossier `photographies_avec_themes/`, puis récupérez la liste de tous les fichiers Turtle dans ce dossier.
Notez que le dossier cible pour enregistrement les fichiers JSONs sera le même, pas besoin donc de distinguer `INPUT_DIR` et `OUTPUT_DIR`.
```python
DIR = Path(__file__).parent  /  "photographies_avec_themes"
turtle_files  =  list(input_dir.glob("*.ttl"))
```

> 🎬 Placez les appels à `import_turtle_file()` et `build_summary_report()`, la déclaration des messages, l'appel au modèle Mistral et l'instruction `print()` finale à l'intérieur d'une boucle qui itère sur chaque fichier de graphe :
```python
for turtle_file in turtle_files:	
	print(f"Traitement de {turtle_file}...")
	data  =  import_turtle_file(turtle_file)
	report  =  build_summary_report(data)
	... # La suite
```
> ⚠️ Attention à l'indentation !	

> 🎬  Testez en exécutant le traitement pour **1 seul graphe** en utilisant le mécanisme de *slicing* 
```python
for turtle_file in turtle_files[:1]:
	... # La suite
```


> 💡 Pour un code plus lisible, vous pouvez déplacer le prompt système dans une constante `SYSTEM_PROMPT` placée en début de script.

> 🎬 En début de script, ajoutez la déclaration de la fonction suivante, qui prend en paramètre le chemin vers le graphe `.ttl` stocké dans la variable de boucle `turtle_file` ainsi que la réponse du modèle Mistral `chat_response` et sauvegarde le résultat en JSON sur le disque dur à coté du fichier ` :
```python 
from  mistralai  import  ChatCompletionResponse
import  json


def  save_to_json(chat_response:  ChatCompletionResponse,  turtle_file:  Path):
	"""Sauvegarde la réponse de Milstra en JSON à coté du fichier `turtle_file`."""
	output_file  =  turtle_file.with_suffix(".json")
	response_content  =  chat_response.choices[0].message.content
	with  open(output_file,  "w",  encoding="utf-8")  as  file:
		json_object  =  json.loads(response_content)
		json.dump(json_object,  file,  ensure_ascii=False,  indent=4)
```
>🎬  Appelez cette fonction juste après l'instruction `print(chat_response.choices[0].message.content)` en lui passant la réponse du modèle et le chemin vers le fichier Turtle du graphe.
```python 
for turtle_file in turtle_files[:1]:
	... # contenu de la boucle
	print(chat_response.choices[0].message.content)
	save_to_json(chat_response,  turtle_file)
```
>🎬 Exécutez de nouveau le script puis **vérifiez** que le dossier `photographies_avec_themes/` contient bien un fichier JSON contenant la hiérarchie extraite pour le graphe choisi !

Dans sa version gratuite, Mistral **impose** une limite de **une requête maximum par seconde**.
Nous ***devons** donc forcer ce délai pour éviter que les requêtes soient rejetées par Mistral.
Une manière simpliste mains fonctionnelle consiste à obliger le script à attendre un certain temps après chaque boucle, grâce à la fonction `sleep(n_seconds)` de la bibliothèque  `time`.

>🎬 Ajoutez l'import de `time` en entête du script :
```python 
import time
```
>🎬 Forcez le script à attendre par exemple 1.5 seconde après avoir enregistré la hiérarchie du graphe courant en JSON, avant de passer au fichier suivant : 
```python 
for turtle_file in turtle_files[:1]:
	... # contenu de la boucle
	print(chat_response.choices[0].message.content)
	save_to_json(chat_response,  turtle_file)
	time.sleep(1.5)
```
>🎬 Vous pouvez maintenant retirer le *slicing* sans crainte puis exécuter enfin l'extraction structurée pour toutes les photographies ! 🥳

>💡 Observez les résultats d'extraction qui s'affichent au fur et à mesure sur le terminal.
> Est-ce que le schéma que l'on a fixé est toujours respecté par le LLM ? 
> Discutons-en ! 💬

###  🏁 Fin du chapitre 2

Félicitations, vous voici équipé avec un script fonctionnel **d'extraction structurée** utilisant un LLM de Mistral ! 🎉

Une fois le traitement effectué sur tous les graphes, chaque fichier `.ttl` devrait avoir son fichier compagnon `.json` décrivant la hiérarchie géographique de la photographie. 

Vous avez maintenant toutes les données utiles pour **géocoder** et **cartographier** le font Atget - ce qu'on verra dans la **partie 3** ! 
