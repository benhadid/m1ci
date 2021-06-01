---
type: lab
date: 2021-06-02T08:00:00+1:00
title: 'Introduction à OpenCL'
attachment: /static_files/labs/lab_04.zip
hide_from_announcements: True
due_event:
    type: due
    date: 2021-06-02T18:00:00+1:00
    description: 'Travaux Pratiques #4 - à remettre'
---

# Objectifs  

  - Vérifier que vous pouvez compiler et exécuter un << kernel >> OpenCL

  - Maitriser les commandes d'exécution de << kernel >> OpenCL et les instructions de gestion des objets mémoires associés

  - Comprendre comment définir les arguments d'exécution pour un << kernel >> OpenCL

  - Comprendre l'interface << hôte / kernel >> OpenCL


# Exercice 1 : (addition de vecteurs)

Commencez par télécharger le fichier de démarrage fourni ci-dessus et décompressez-le dans le répertoire de votre choix.

Dans cet exercice, vous allez simplement compiler puis exécuter le programme OpenCL fourni qui additionne deux tableaux A et B; et retourne le résultat dans un tableau C. Consultez le code hôte dans `vadd_c.c` et identifiez les appels d'API OpenCL discutés en cours.

Pour compiler puis exécuter le programme, utilisez les commandes 

```bash
$ make vadd
$ ./vadd
```

## Résultat attendu :
 - Un message vérifiant que l'addition des vecteurs s'est terminée avec succès


# Exercice 2 : (série d'addition de vecteurs)

Dans cet exercice, vous allez modifier le fichier `vadd_c.c` fourni dans l'exercice n° 1 afin de réaliser plusieurs opérations d'addition de tableaux.

  - Ajoutez des objets mémoires supplémentaires et assignez-les à des vecteurs définis sur l'hôte (voir le programme vadd fourni pour des exemples)
  - Codez une série d'opérations d'addition de vecteurs… par ex. C = A + B; D = C + E; F = D + G.
  - Récupérez le résultat final et vérifiez qu'il est correct

Pour compiler puis exécuter le programme, utiliser les mêmes commandes fournies dans l'exercice n° 1

## Résultat attendu :
 - Un message vérifiant que la série d'addition de vecteurs s'est terminée avec succès


# Exercice 3 : (addition de trois vecteurs)

Dans les exercices précédents, le *kernel* fourni permet d'additionner deux tableaux à la fois. Nous allons maintenant modifier notre programme (hôte et kernel) afin de réaliser une opération d'addition de trois tableaux en un coup.

 - Modifiez le << kernel >> dans `vadd_c.c` pour additionner trois tableaux en un coup (ex. D = A + B + C)
 - Modifiez ensuite le code hôte pour définir trois vecteurs en entrée et associez-les aux arguments du << kernel >> respectifs.
 - Récupérez le résultat final et vérifiez qu'il est correct

Pour compiler puis exécuter le programme, utiliser les mêmes commandes fournies dans l'exercice n° 1

