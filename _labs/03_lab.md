---
type: lab
date: 2022-04-07T08:00:00+1:00
title: 'OpenMP et instructions vectorielles'
attachment: /static_files/labs/lab_03.zip
hide_from_announcements: True
due_event:
    type: due
    date: 2022-04-17T18:00:00+1:00
    description: 'Travaux Pratiques #3 - à remettre'
---

# Objectifs  

  - Découvrir le parallélisme des données avec les données vectorielles

  - Optimiser le code en utilisant OpenMP et les instructions intrinsèques

  - Optimiser le code en utilisant OpenMP, les instructions intrinsèques et le << cache blocking >>


# Exercice 1 : (calcul du produit matricielle C = A * B - version Séquentielle)

Commencez par télécharger le fichier de démarrage fourni ci-dessus et décompressez-le dans le répertoire de votre choix.

Dans cet exercice, implémentez dans le fichier `matmul_v1.c` la section indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN** afin d'effectuer, de façon **séquentielle**, le produit des matrices `A` et `B` et sauvegarder le résulat dans la matrice `C`.

Une fois votre implémentation faite, vous pouvez compiler puis exécuter le programme à l'aide des commandes 

```bash
$ make v1
$ ./v1
```

Prenez note du temps d'exécution retourné pour comparaison avec les exercices suivants

# Exercice 2 : (calcul du produit matricielle C = A * B - version vectorisée)

En vous appuyant sur les slides du cours , complétez la partie indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN** dans le fichier
`matmul_v2.c` afin d'éffectuer la même tâche que précédemment, c-à-d. multiplier les matrices `A` et `B` et stocker le résultat dans la matrice `C`; sauf que cette fois-ci, utiliser les instructions intrinsèques pour implémenter une version << vectorisée >> du programme.

Pour compiler puis exécuter cette version du programme lancez les commandes 

```bash
$ make v2
$ ./v2
```

Comparez le temps d'exécution retourné ici avec la valeur retournée dans l'exércice 1. Votre nouvelle version est-elle plus rapide ou plus lente ?

# Exercice 3 : (calcul du produit matricielle C = A * B - version OpenMP)


Dans cet exercice, vous allez implémenter la version parallèle du produit matricielle codé dans l'éxercice 1. En sens, reprenez votre code de l'exercice 1 et insérer le dans la partie indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN**  dans le fichier `matmul_v3.c`. Apportez ensuite les modifications nécessaires pour paralléliser votre code avec **OpenMP**.

Pour compiler puis exécuter cette version du programme lancez les commandes 

```bash
$ make v3
$ ./v3
```

Comparez le temps d'exécution retourné ici avec les valeurs retournées dans les exércices  précédents. Que remarquez-vous ? 


# Exercice 4 : (calcul du produit matricielle C = A * B - vectorisation et OpenMP)

Dans cet exercice, vous allez implémenter la version **parallèle et vectorisée** du produit matricielle codé dans les éxercice 1 & 2. En sens, reprenez votre code de l'exercice précédent et insérer le dans la partie indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN**  dans le fichier `matmul_v4.c`. Apportez ensuite les modifications nécessaires pour *vectoriser* votre code avec des instructions intrinsèques.

Pour compiler puis exécuter cette version du programme lancez les commandes 

```bash
$ make v4
$ ./v4
```

Comparez le temps d'exécution retourné ici avec les valeurs retournées dans les exércices précédents. Que remarquez-vous ? 


# Exercice 5 : (calcul du produit matricielle C = A * B - vectorisation + OpenMP + cache blocking)

Afin d'améliorer la quantité de réutilisation des données dans les caches pendant le produit de matrice, nous pouvons utiliser la technique du << cache-blocking >> déjà rencontrée dans l'exercice 2 du [premier]({{site.baseurl}}/labs/01_lab) TP. Reprenez votre code de l'exercice précédent
et insérer le dans la partie indiquée par les marqueurs **BLOC_DEBUT** et **BLOC_FIN**  dans le fichier `matmul_v5.c`. Apportez ensuite les modifications nécessaires au code pour implémenter le produit par bloc des matrices `A` et `B`.

Pour compiler puis exécuter cette version du programme lancez les commandes 

```bash
$ make v5
$ ./v5
```

Comparez le temps d'exécution retourné ici avec les valeurs retournées dans les exércices précédents.


