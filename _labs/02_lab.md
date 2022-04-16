---
type: lab
date: 2022-04-14T12:00:00+1:00
title: 'Le multitâche (Threads)'
attachment: /static_files/labs/lab_02.zip
#solutions: /static_files/labs/lab_solutions.pdf
hide_from_announcements: False
due_event:
    type: due
    date: 2022-04-17T18:00:00+1:00
    description: 'Travaux Pratiques #2 - à remettre'
---

# Objectifs  

  - Découvrir le parallélisme de programme avec les threads POSIX

  - Apprendre à correctement gérer les << sections critiques >> dans un programme

  - Découvrir le parallélisme de programme avec OpenMP


# Exercice 1 : (retourner le minimum dans un tableau - version Séquentielle)

Commencez par télécharger le fichier de démarrage fourni ci-dessus et décompressez-le dans le répertoire de votre choix.

Dans cet exercice, implémentez dans le fichier `min_sequentiel.c` le corp de la fonction`min()`
(indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN**)  qui effectue une recherche séquentielle puis retourne le minimum dans un tableau d'entiers (voir figure ci-dessous).

![minimum]({{site.baseurl}}/static_files/labs/lab02/sequentielle.jpg){: .aligncenter width="50%" height="50%" }     

Ici, la valeur qui doit être retournée par le programme est égale à << **1** >>

Une fois votre implémentation faite, vous pouvez compiler puis exécuter le programme à l'aide des commandes 

```bash
$ make ex1
$ ./v1
```
Prenez note du temps d'exécution retourné pour comparaison avec les exercices suivants

# Exercice 2 : (retourner le minimum dans un tableau - version POSIX)

En vous appuyant sur les slides du cours et l'implémentation fournie dans le fichier `min_posix.c`, complétez le corp de la fonction `min_thread()` entre les marqueurs **BLOC_DEBUT** et **BLOC_FIN**  afin d'effectuer la même tâche que précédemment, c-à-d. une recherche de la valeur minimum dans un tableau; sauf que cette fois-ci, votre boucle doit s'exécuter en parallèle.

![minimum_posix]({{site.baseurl}}/static_files/labs/lab02/posix.jpg){: .aligncenter width="40%" height="40%" } 

N'oubliez pas d'introduire les instructions nécessaires pour gérer correctement les << sections critiques >> s'il y a lieu. 

Pour compiler puis exécuter cette version du programme lancez les commandes 

```bash
$ make ex2
$ ./v2
```

Comparez le temps d'exécution retourné ici avec la valeur retournée dans l'exercice 1. Votre nouvelle version est-elle plus rapide ou plus lente ? (**Indication** : si implémentée correctement, la version parallélisée devrait être plus rapide que la version séquentielle).

# Exercice 3 : (retourner le minimum dans un tableau - version OpenMP)

Dans cet exercice, vous allez implémenter entre les marqueurs **BLOC_DEBUT** et **BLOC_FIN**  dans le fichier `min_openmp.c`, la version parallèle **OpenMP** du programme de recherche de la valeur minimum dans un tableau. Comparée à la version POSIX de l'exercice n°2, la version **OpenMP** devrait être plus simple à mettre en oeuvre.

Pour compiler puis exécuter cette version du programme lancez les des commandes 

```bash
$ make ex3
$ ./v3
```

Comparez le temps d'exécution retourné ici avec les valeurs retournées dans les exercices  précédents. Que remarquez-vous ? 
