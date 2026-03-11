import os
from mistralai import Mistral, ChatCompletionResponse
from fetch_themes import build_summary_report, import_turtle_file
from pathlib import Path
import time
import json

"""
Concernant les commentaire de cette fonction : 
C'est une fonction donnée par le prof, mais il ne l'explique pas exactement et je ne comprenais pas tout donc j'ai demandé à Mistral de la commenter
Les commentaire suivants sont donc une simplification et reformulation de ce que j'ai compris des commentaires de Mistral
"""
def save_to_json(chat_response: ChatCompletionResponse, turtle_file: Path): #Chat response : paramètre typé qui suggère l'utilisation d'une API (provient sûrement du module mistralai)
    """Enregistre la réponse de l'API dans un fichier JSON."""
    output_file = turtle_file.with_suffix(".json") #output_file indique le fichier de sortie. ".with_suffix" remplace l'extension du fichier. Ici de .ttl à .json. Évite les manipulations manuelles de chaînes de caractères pour gérer les extensions.
    response_content = chat_response.choices[0].message.content #La réponse que l'on obtient est le premier choix proposé par le chat. 
    with open(output_file, "w", encoding="utf-8") as file: #Ouverture du fichier json. "w" signifie "write", donc nous voulons écrire dans notre fichier json. Si il n'existe pas, il est crée. 
        json_object = json.loads(response_content) #Convertit la chaine Json en objet Python. json.loads() convertit la chaine de caractère json (ici response_content) en objet python afin de le rendre manipulable (sûrment un dictionnaire un truc du genre).
        json.dump(json_object, file, ensure_ascii=False, indent=4) #Ecrit l'objet Python dans le fichier. ensure_ascii permet de garder les caractère "non-ASCII", type accents. indent ajout un indent (Tu t'y attendais pas à celle là.)

api_key = os.environ["MISTRAL_API_KEY"] 
"""
Importe la clef API qui est stockée dans l'environnement. 
Nous n'avons pas de .env comme nous n'utilisons pas venv mais uv. Nous l'avons stockée en utilisant dans le terminal "export MISTRAL_API_KEY=clefAPI". 
Il n'est pas correct de laisser une clef API visible en ligne, c'est pourquoi nous la cachons.
"""
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

DIR=Path(__file__).parent / "photographies_avec_themes" #Chemin vers le dossier traité
turtle_files=list(DIR.glob("*.ttl")) # Liste tous les fichiers dans ce dossier

for turtle_file in turtle_files:
    print(f"Processing {turtle_file}...") #Imprime quel fichier est traité
    # Appelle les fonctions crées dans fetch_themes.py
    data = import_turtle_file(turtle_file)
    report = build_summary_report(data)

    messages = [
        {
            "role": "system", #Role system : Ce que le LLM va garder en mémoire à chaque itération.
            #Pour "content" il nous demande de créer une variable ailleurs mais j'ai pas envie, je comprend mieux comme ça
            "content": """
            # Rôle
            Tu es un expert en extraction d'information géographique dans des métadonnées patrimoniales.

            # Tâche
            À partir d'un résumé descriptif d'une photographie ancienne, tu dois :
            1. identifier les entités géographiques qui renseignent sur **la localisation du sujet de la photographie** dans l'espace ;
            2. lister ces entités les uns après les autres.

            # Règles
            - les entités doivent être triée de la plus précise à la plus générale.
            - si aucune ville n'est mentionnée, la ville est Paris.
            - si aucun pays n'est mentionné, le pays est la France.

            # Exemple
            **Résumé descriptif **
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

            **Réponse JSON**
            {
                "toponyme": "Au Soleil d'Or",
                "adresse": "84 rue Saint-Sauveur",
                "voie" : "rue Saint-Sauveur",
                "ville": "Paris",
                "pays": "France"
            }


            Le résumé à traiter sera donné dans le prochain input.
            """,
        },
        {
            "role":"user", # user se renouvelle à chaque demande. 
            "content": report, #Message demandé à L'IA en gros.
        }
    ]
    chat_response = client.chat.complete(
        model = model, #Définit le modèle de langage (c'est la version en gros)
        messages = messages, #Liste des messages à envoyer (définit précédemment)
        response_format = {
            "type": "json_object", #On demande la réponse sous format json
        }
    )

    print(chat_response.choices[0].message.content) # Réponse du modèle 
    save_to_json(chat_response, turtle_file) 
    time.sleep(1.5)  # Pause pour éviter de surcharger l'API