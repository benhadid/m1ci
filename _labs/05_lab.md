---
type: lab
date: 2021-06-02T08:00:00+1:00
title: 'Programmation avancée avec OpenCL'
attachment: /static_files/labs/lab_05.zip
hide_from_announcements: True
due_event:
    type: due
    date: 2021-06-02T18:00:00+1:00
    description: 'Travaux Pratiques #5 - à remettre'
---

# Objectifs  

  - Familiarisation avec le workflow de développement OpenCL.

  - Optimisation de calcul sur une plateforme OpenCL.

# Exercice 1

Ce TP reprend les concepts discutés en cours sur les techniques d'optimisation de calcul dans OpenCL. En ce sens, vous implémenterez les algorithmes de multiplication de matrice carrée (A[N][N] * B[N][N] = C[N][N]) discutés en cours en utilisant OpenCL et en étudiant leurs performances sur GPU. 

La mise en œuvre comprend deux versions :

  - les matrices A et B sont stockées dans la mémoire globale et le kernel de calcul accède aux données directement à partir de la mémoire globale

  - les matrices A et B sont chargées dans la mémoire locale d'un groupe de threads (c.-à-d. un work-group) et le kernel de calcul accède aux données dans cette mémoire locale. Ici, le kernel réalise une multiplication par bloc des matrices A et B et effectue une sommation partielle dans C pour obtenir le résultat final (voir les slides du cours ...).
  
Un code squelette est déjà fourni dans le kit de démarrage pour vos implémentations. Veuillez vous référer aux slides du cours << [OpenCL (les bases)](https://1drv.ms/p/s!Agf0g-qZKM8_4hhj1SAcibJZJX3g?e=pnA2Xh) >> pour la mise en œuvre des deux versions. Le slide n° 7 fournit la mise en œuvre de la version 1 et les slides n° 39 à 42 pour la version 2.

Les algorithmes des versions 1 et 2 auront besoin d'une topologie en deux dimensions pour les threads dans un work-group; même observation peut être faite pour la topologie des groupes de threads dans la grille de calcul. Choisissez 16 x 16 pour la taille du work-group. Pour simplifier, vous pouvez supposer que la taille de la matrice N est un nombre multiple de 16 ( 16 = dimension_horizontale(work_group) = dimension_verticale(work_group) ).

Pour compiler puis exécuter le programme, entrez les commandes suivantes :

```bash
$ make
$ ./mult
```


# Exercice 2

En reprenant la deuxième version du produit de matrices effectué dans l'exercice précédent, opérez les changements nécessaires dans votre code pour réaliser des multiplications de matrices pas nécessairement carrées : (A[M][K] * B[K][N] = C[M][N]) avec M ≠ N. Les topologies des threads dans un work-group; et des groupes de threads dans la grille de calcul restent unchangés.

Dans une première version, vous pouvez supposer que M et N sont des multiples de 16 (comme pour l'exercice précédent). 

Dans une version un peu plus compliquée, votre programme devrait réaliser des multiplications de matrices de tailles qui ne sont pas nécessairement des multiples de 16 (dimensions quelconques). 
