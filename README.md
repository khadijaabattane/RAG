# RAG Contrats MAIF

Une démonstration de pipeline RAG pour interroger et générer des réponses à partir de documents PDF (contrats, législation, etc.).



## Table des matières

1. [Description du projet](#description-du-projet)
2. [Fonctionnalités](#fonctionnalités)
3. [Architecture](#architecture)
4. [Prérequis](#prérequis)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Utilisation](#utilisation)

   * [1. Construction de l’index](#1-construction-de-lindex)
   * [2. Lancement de l’application Streamlit](#2-lancement-de-lapplication-streamlit)
   * [3. Exemple de notebook](#3-exemple-de-notebook)
8. [Structure du projet](#structure-du-projet)




## Description du projet

Ce projet met en œuvre un pipeline RAG  permettant de :

* Charger un corpus de documents PDF.
* Découper les textes (`chunking`) et métadonnées (ID, page, etc.).
* Embedding de chaque chunk via un modèle de vecteurs (paraphrase-multilingual-MiniLM-L12-v2).
* Construction d’un index vectoriel FAISS.
* Récupération des chunks pertinents pour une requête utilisateur.
* Génération de réponses contextualisées avec un LLM (Mistral-7B).


## Fonctionnalités

* **Loader PDF** : lecture de tous les PDF d’un dossier.
* **Chunker** : découpage en unités textuelles avec overlap.
* **Embedder** : appel à un service d’embeddings pour vectoriser les chunks.
* **Vector Store** : index FAISS gérant embeddings et métadonnées.
* **RAG Retriever** : recherche des chunks les plus similaires.
* **Generator** : génération de texte via LLM.
* **Streamlit App** : interface web simple pour tester des requêtes.
* **Notebook démo** : pipeline end-to-end commenté.

## Architecture

```
rag/
├── core/          # Modules du pipeline
│   ├── config.py         # Chemins et constantes
│   ├── loader.py         # Chargement des PDFs
│   ├── chunker.py        # Découpage en chunks
│   ├── embedder.py       # Génération d'embeddings
│   ├── vector_store.py   # Index FAISS
│   ├── retriever.py      # Récupération RAG
│   └── generator.py      # Appel LLM 
├── scripts/    # Scripts (build_index)
├── app/        # Application Streamlit
├── notebooks/     # Notebooks de démonstration
├── data/          # Répertoire des PDFs source
└── README.md  
└── requirements.txt  
└── setup.py      # pour Packaging  
```

## Prérequis

* Python 3.8+
* pip
* (Optionnel) Un environnement virtuel (venv, conda)

## Installation

1. Cloner le dépôt :

   ```bash
   git clone git@github.com:khadijaabattane/RAG.git
   cd rag
   ```
2. Créer et activer un environnement virtuel :

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .\.venv\Scripts\activate  # Windows
   ```
3. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```
4. Installer le package en mode editable :

   ```bash
   pip install -e .
   ```

## Configuration

* Copier et éditer `core/config.py` pour définir :

  * `PDF_DIR` : chemin vers le dossier contenant les PDF.
  * `INDEX_PATH` : où sauvegarder l’index FAISS.
  * `METADATA_PATH` : où sauvegarder les métadonnées.
  * Clé API pour le service d’embeddings / LLM.

## Utilisation

### 1. Construction de l’index

```bash
python -m rag.scripts.build_index
```

### 2. Lancement de l’application Streamlit

```bash
export PYTHONPATH=$(pwd)
python -m streamlit run rag/app/app.py
```

Puis ouvrez l'application en localhost dans votre navigateur.

### 3. Exemple de notebook

Un notebook prêt à l’emploi se trouve dans `notebooks/demo_pipeline.ipynb` et illustre chaque étape du pipeline.

## Structure du projet

* **core/** : code métier du pipeline RAG.
* **scripts/** : scripts CLI.
* **app/** : application Streamlit.
* **notebooks/** : Jupyter notebooks de démo.
* **data/** : exemples de PDF.
* **requirements.txt** : dépendances Python.


