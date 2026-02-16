"""
Ce script a pour objectif d'enrichir les graphes RDF des photographies
avec les labels des thèmes Rameau associés à ces photographies.
"""

# ----------------------------------------------------------------------------------------------------------------------------------------
# --- A. IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------------------

# Annotations de type
from typing import Iterable

# os et pathlib pour la gestion des fichiers et des chemins
import os
from pathlib import Path

# Modules et fonctions utiles pour interagir avec les graphes RDF et le point d'accès SPARQL de la BnF
from SPARQLWrapper import JSONLD, SPARQLWrapper
from rdflib.namespace import DCTERMS
from rdflib import RDF, SKOS, Graph, Node, URIRef
from utils import save_graph_html

# ----------------------------------------------------------------------------------------------------------------------------------------
# --- B. CONFIGURATION GLOBALE
# ----------------------------------------------------------------------------------------------------------------------------------------

# Point d'accès SPARQL de la BnF pour interroger les données Rameau
DATA_BNF_ENDPOINT = "https://data.bnf.fr/sparql"

# ----------------------------------------------------------------------------------------------------------------------------------------
# --- C. FONCTIONS PRINCIPALES
# ----------------------------------------------------------------------------------------------------------------------------------------


def setup_bnf_sparql_wrapper() -> SPARQLWrapper:
    """
    Crée un objet SPARQLWrapper pour interroger le point d'accès SPARQL de la BnF.

    Returns:
        SPARQLWrapper: Un objet SPARQLWrapper configuré.
    """
    endpoint = SPARQLWrapper(DATA_BNF_ENDPOINT)
    endpoint.setTimeout(60)
    return endpoint


def import_turtle_file(turtle_file: Path) -> Graph:
    """
    Charge un graphe RDF à partir d'un fichier Turtle sur le disque.

    Args:
        turtle_file (Path): Le chemin vers le fichier Turtle à charger.

    Returns:
        Graph: Un objet Graph contenant les données RDF chargées depuis le fichier.
    """
    g = Graph()
    g.parse(turtle_file, format="turtle")
    return g


def identify_photo_resource(graph: Graph) -> Node:
    """
    Extrait la ressource de type Manifestation décrivant la photo.
    Le prédicat utilisé est RDA:manifestation (http://rdaregistry.info/Elements/c/#C10007)
    Args:
        graph (Graph): Le graphe RDF à partir duquel extraire l'URI de la photo.
    Returns:
        Node: Le noeud du graphe RDF correspondant à la ressource de type RDA:manifestation décrivant la photo.
    """
    photo_uri = graph.value(
        predicate=RDF.type, object=URIRef("http://rdaregistry.info/Elements/c/#C10007")
    )
    if not photo_uri:
        raise ValueError("Aucune ressource de type RDA:manifestation dans le graphe.")
    return photo_uri


def get_rameau_themes(graph: Graph) -> list[Node]:
    """
    Retourne les sujets associés à une photographie dans son graphe.
    Les sujets sont liés à la photographie via la propriété DCTERMS.subject :
    [ sujets Rameau ] -- DCTERMS.subject --> [ photographie ]

    Args:
        graph (Graph): Le graphe RDF d'une photographie.
    Returns:
        list[Node]: Une liste de noeuds du graphe RDF correspondant aux sujets Rameau associés à la photographie.
    """
    manifestation = identify_photo_resource(graph)
    return list(graph.objects(subject=manifestation, predicate=DCTERMS.subject))


def fetch_themes_labels(themes: Iterable, databnf: SPARQLWrapper) -> Graph:
    """
    Interroge le point d'accès SPARQL de la BnF pour récupérer les labels des thèmes Rameau.
    Args:
        themes (Iterable): Les URIs des thèmes Rameau à interroger.
        databnf (SPARQLWrapper): L'objet SPARQLWrapper configuré pour interroger le point d'accès SPARQL de la BnF.
    Returns:
        Graph: Un graphe RDF contenant les labels des thèmes récupérés depuis la BnF
    """

    if not themes:
        print("Aucun thème Rameau à interroger.")
        return Graph()

    # On a pas besoin de tous les triples associés aux thèmes, mais uniquement de leurs label,
    # qui leurs sont associés via les propriétés SKOS : altLabel et prefLabel :
    # http://www.w3.org/2004/02/skos/core#prefLabel # "preferred label" : le label principal d'un thème.
    # http://www.w3.org/2004/02/skos/core#altLabel # label alternatif.

    # Concatèner les URIs des thèmes Rameau dans une chaîne de caractères pour
    # les injecter dans la requête SPARQL tous ensemble  et éviter de faire une requête SPARQL par thème.
    themes_formatted = " ".join([t.n3() for t in themes])

    # Une requête SPARQL de type CONSTRUCT
    # retourne un graphe RDF construit à partir des résultats de la clause WHERE.
    query = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{
            VALUES ?s {{ {themes_formatted} }}
            VALUES ?p {{ skos:prefLabel skos:altLabel }}
            ?s ?p ?o .
        }}
    """

    print(f"Requête SPARQL construite :\n{query}")

    databnf.setQuery(query)

    # Forcer le format de retour en JSON-LD
    # pour récupérer un objet Graph directement à partir de la réponse de la requête SPARQL.
    databnf.setReturnFormat(JSONLD)

    results = databnf.queryAndConvert()

    # Garde-fou si jamais queryAndConvert change et ne retourne plus un objet Graph,
    # ou si le format de retour n'est plus JSON-LD ou RDF/XML
    if not isinstance(results, Graph):
        results = Graph().parse(data=results, format="json-ld")  # type: ignore

    return results


def build_summary_report(graph: Graph) -> str:
    """
    Construit un rapport de synthèse des thèmes associés à une photographie à partir de son graphe RDF.
    Args:
        graph (Graph): Le graphe RDF d'une photographie.
    Returns:
        str: Une chaîne de caractères contenant le rapport de synthèse.
    """
    # 1. Récupère l'URI de la photo et son titre
    photo_uri = graph.value(
        predicate=RDF.type, object=URIRef("http://rdaregistry.info/Elements/c/#C10007")
    )
    titre = graph.value(subject=photo_uri, predicate=DCTERMS.title)

    # 2. Prépare l'en-tête de l'infobox
    report = f"\n=== PHOTO : {titre} ===\n"
    report += f"Lien : {photo_uri}\n"
    report += "Thèmes assignés:\n"

    # 3. Pour chaque thème associé à la photo...
    for theme in graph.objects(subject=photo_uri, predicate=DCTERMS.subject):

        # ... on affiche le label principal du thème
        pref = graph.value(subject=theme, predicate=SKOS.prefLabel)
        report += f" • « {pref} »"

        # ... puis les alternatives
        altLabels = list(graph.objects(subject=theme, predicate=SKOS.altLabel))
        if altLabels:
            report += f" - altLabels : {', '.join(f'« {label} »' for label in altLabels)}"  # type: ignore
        report += "\n"

    # 4. Retourne le rapport de synthèse construit
    return report


def merge_labels_into_photo_graph(
    photo_graph: Graph, rameau_labels_graph: Graph
) -> Graph:
    """
    Ajoute les labels des thèmes Rameau au graphe RDF d'une photographie.
    Args:
        photo_graph (Graph): Le graphe RDF d'une photographie.
        rameau_labels_graph (Graph): Le graphe RDF contenant les labels des thèmes Rameau récupérés depuis la BnF.
    Returns:
        Graph: Le graphe RDF de la photographie enrichi avec les labels des thèmes Rameau.
    """

    return photo_graph + rameau_labels_graph


def export_to_turtle(graph: Graph, output_file: Path) -> None:
    """
    Enregistre un graphe RDF sur le disque au format Turtle.
    Args:
        graph (Graph): Le graphe RDF à enregistrer.
        output_file (Path): Le chemin vers le fichier Turtle où enregistrer le graphe.
    Returns:
        None: Cette fonction enregistre le graphe sur le disque et ne retourne rien.
    """
    graph.serialize(destination=output_file, format="turtle")


def export_graph_to_html(graph: Graph, output_file: Path) -> None:
    """Exporte un graphe RDF au format HTML pour visualisation."""
    save_graph_html(graph, output_file.as_posix(), height="600px", notebook=False)


# ----------------------------------------------------------------------------------------------------------------------------------------
# --- D. POINT D'ENTRÉE DU SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__": # permet d'indiquer à l'interpréteur python que cette partie n'est à exécuter que lorsque l'on appelle le fichier comme un script
    """
    POINT D'ENTRÉE DU SCRIPT
    Ce bloc orchestre l'enrichissement des données : il lit les fichiers locaux,
    va chercher les informations manquantes à la BnF, et fusionne le tout.
    """

    # --- ÉTAPE 1 : RÉGLAGES ET CHEMINS (CONFIGURATION)
    base_dir = Path(__file__).parent
    input_dir = base_dir / "photographies" # dossier contenant les fichiers d'origine
    output_dir = base_dir / "photographies_avec_themes" # dossier où les fichiers créés par le script seront envoyés

    # Sécurité : limiter le nombre de requêtes pour les tests (None pour tout traiter)
    graph_processing_limit = 1

    # --- ÉTAPE 2 : PRÉPARATION DES OUTILS (INITIALISATION)

    # On s'assure que le dossier de sortie existe pour ne pas faire planter le script
    os.makedirs(output_dir, exist_ok=True)

    # Connexion au point d'accès SPARQL de la BnF
    databnf = setup_bnf_sparql_wrapper()

    # On récupère la liste de tous les fichiers .ttl à traiter
    turtle_files = list(input_dir.glob("*.ttl")) # 'glob' permet de faire une regex pour récupérer tous les fichiers avec l'extension '.ttl'

    # Application de la limite si elle est définie
    if graph_processing_limit: # si une limite est annoncée dans 'graph_processing_limit' (ligne 231)
        turtle_files = turtle_files[:graph_processing_limit] # [chiffre:chiffre] permet de découper une liste. On s'en sert ici pour couper la liste en partant de 0 à la limite annoncée

    print(f"{len(turtle_files)} fichiers Turtle à traiter dans {input_dir}.") # print pour vérifier combien de fichiers seront traités

    # --- ÉTAPE 3 : BOUCLE PRINCIPALE DE TRAITEMENT

    for turtle_photo_file in turtle_files: # boucle permettant de lancer l'ensemble

        print(f"Début de l'enrichissement pour : {turtle_photo_file.name}") # print d'étape pour indiquer que le code commence à s'exécuter (permet de mesurer la progression)

        try: # essaie d'exécuter le code
            
            # 1. IMPORT : On transforme le fichier texte en un graphe RDF manipulable
            photo_graph = import_turtle_file(turtle_photo_file)

            # 2. IDENTIFICATION : On trouve les ressources du graphe qui
            # correspondent aux thèmes Rameau associés à la photo
            rameau_themes = get_rameau_themes(photo_graph)

            # 3. RÉCUPÉRATION : On demande à la BnF les labels textuels de ces thèmes Rameau.
            rameau_labels_graph = fetch_themes_labels(rameau_themes, databnf)

            # 4. FUSION : On injecte les labels récupérés dans le graphe de la photo.
            enriched_photo_graph = merge_labels_into_photo_graph(
                photo_graph, rameau_labels_graph
            )

            # 5. CONTRÔLE : On affiche un résumé textuel dans la console
            report = build_summary_report(enriched_photo_graph)
            print(report)

            # 6. EXPORT : On sauvegarde le résultat final sur le disque
            enriched_photo_file = output_dir / turtle_photo_file.name
            export_to_turtle(enriched_photo_graph, enriched_photo_file)

            # Bonus : on exporte aussi une version HTML du graphe enrichi
            # pour pouvoir le visualiser facilement dans un navigateur
            export_graph_to_html(
                enriched_photo_graph, enriched_photo_file.with_suffix(".html") # change l'extension du fichier
            )

            print(f"Réussite : {turtle_photo_file.name} enrichi.") # print pour indiquer que le script a abouti, graph par graph

        except Exception as e: # si l'exécution échoue pour un graph, affiche un message d'erreur sans arrêter l'exécution du script
            print(f"Échec pour {turtle_photo_file.name} : {e}")