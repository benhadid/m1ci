---
type: lab
date: 2022-03-24T12:00:00+1:00
title: 'Mémoire Cache'
attachment: /static_files/labs/lab_01.zip
#solutions: /static_files/labs/lab_solutions.pdf
hide_from_announcements: False
due_event:
    type: due
    date: 2022-04-10T18:00:00+1:00
    description: 'Travaux Pratiques #1 - à remettre'
---

# Objectifs  

  - Découvrir comment les schémas d'accès à la mémoire déterminent les taux de succès de cache.

  - Déterminer quels schémas d'accès à la mémoire produisent de BONS taux de succès.

  - Être en mesure d'optimiser le code pour produire de bons taux de succès de cache.


# Exercice 1 : (Ordre des boucles et multiplication matricielle)

Pour rappel, les matrices sont des structures de données à deux dimensions dans lesquelles chaque élément de données est accessible via deux indices. Pour multiplier deux matrices, nous pouvons simplement utiliser trois boucles imbriquées. Par exemple, en supposant des matrices A, B et C de dimensions n-par-n et stockées dans des tableaux de colonnes à une dimension :

```C
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            C[i+j*n] += A[i+k*n] * B[k+j*n];
```

Les opérations de multiplication matricielle sont au cœur de nombreux algorithmes d'algèbre linéaire, et une multiplication matricielle efficace est essentielle pour de nombreuses applications dans les sciences appliquées.

Dans le code ci-dessus, notez que les boucles sont ordonnées i, j, k. Si nous examinons la boucle la plus interne (celle qui incrémente k), on voit qu'elle...

 -	accède le tableau B avec un pas de 1
 -	accède le tableau A avec un pas de n
 -	accède le tableau C avec un pas de 0 (ne dépend pas de k)

Pour calculer **correctement** la multiplication matricielle, l'ordre des boucles n'a pas d'importance. **MAIS**, l'ordre dans lequel nous choisissons d'accéder aux éléments des matrices peut avoir **un impact important sur les performances**. Les caches fonctionnent mieux (i.e. un meilleur taux de succès) lorsque les accès à la mémoire exposent une localité spatiale et temporelle, permettant la réutilisation des blocs de données déjà contenus dans le cache. L'optimisation des schémas d'accès à la mémoire dans un programme est essentielle pour obtenir de bonnes performances.

Ouvrez le fichier en langage C `matrixMultiply.c` dans l'éditeur de votre choix et examinez son contenu. Vous remarquerez que le fichier contient six implémentations (l'une d'elles est illustrée ci-dessus) de multiplication de matrices en utilisant des ordres différents des trois boucles imbriquées.

**Tâche :** Déduisez les pas utilisés dans chaque ensemble de boucles imbriquées des cinq autres implémentations.

Compilez et exécutez le fichier `matrixMultiply.c` avec la commande suivante, puis répondez aux questions ci-dessous.

```bash
$ make ex2
```

Notez que la commande de compilation dans le Makefile utilise l'indicateur '-O3'. Ce paramètre permet d'activer toutes les optimisations de performance possible du compilateur. La commande ci-dessous exécutera quelques multiplications de matrice selon les six implémentations différentes dans le fichier, et affichera la vitesse à laquelle chaque implémentation a exécuté l'opération. L'unité << Gflops/s >> signifie : << Giga-opérations en virgule flottante par seconde >>. Plus le nombre est grand, plus le calcul est rapide !

1. Quel(s) ordre(s) de boucles donne(nt) le meilleur résultat ? Pourquoi ?

2. Quel(s) ordre(s) de boucles donne(nt) le pire résultat ? Pourquoi ?

3. Comment la façon dont nous parcourons les matrices affecte-t-elle les performances ?


# Exercice 2: (Transposition de matrice par bloc)

## Transposition matricielle

Nous souhaitons permuter les lignes et les colonnes d'une matrice (voir figure ci-dessous). Cette opération est appelée *transposition de matrice* et une implémentation efficace peut être très utile, particulièrement quand on effectue des opérations assez compliquées en algèbre linéaire. La transposée de la matrice A est souvent désignée par A<sup>*T*</sup>.

![Transposition]({{site.baseurl}}/static_files/images/matrix_transpose.png){: .aligncenter width="50%" height="50%" }     

## Le &laquo; cache-blocking &raquo;

Dans l'exercice précédent sur les multiplications de matrices, nous parcourons (avec des pas différents) toutes les valeurs des matrices A et B pour calculer une valeur de la matrice C. Ainsi, nous accédons constamment à de nouvelles valeurs de la mémoire et obtenons très peu de localité temporelle et / ou spatiale des accès mémoire !

Nous pouvons améliorer la quantité de réutilisation des données dans les caches en implémentant une technique appelée << cache-blocking >>. Plus formellement, Le << cache-blocking >> est une technique qui consiste à ré-écrire une opération sur les tableaux de sorte à forcer la réutilisation des données présentes dans le cache. Elle doit donc prendre en compte la taille du cache comme argument. Dans le cas de la transposition matricielle, on envisage d'effectuer la transposition un bloc à la fois.

![BlocTransposition]({{site.baseurl}}/static_files/images/block_matrix_transpose.png){: .aligncenter width="50%" height="50%"}     

Dans l'image ci-dessus, nous transposons chaque sous-matrice $$A_{ij}$$ de la matrice $$A$$ dans son **emplacement final** dans la matrice de sortie, une sous-matrice à la fois. Nous pouvons vérifier que la transposition de chaque sous-matrice individuelle est équivalent à la transposition de la matrice entière.

Puisque la transposition de la matrice entière est effectuée une sous-matrice à la fois, cela permet de consolider en cache les accès mémoire à ce petit morceau de données lors de la transposition de cette sous-matrice particulière; ce qui augmente le degré de localité temporelle (et spatiale) que nous exposons et améliore ainsi les performances.

Dans cet exercice, vous allez compléter une implémentation pour la transposition de matrice et analyser ses performances.
En particulier, votre tâche consiste à implémenter la technique du << cache-blocking >> dans la fonction `transpose_blocking()` dans le fichier `transpose.c`. **Vous ne devez PAS supposer que la largeur de la matrice (`n`) est un multiple de la taille du bloc `blocksize`**. Après avoir implémenté la fonction `transpose_blocking()`, vous pouvez compiler et exécuter votre code en entrant sur la console la commande :

```bash
$ make ex3
$ ./transpose n blocksize
```

où `n` est la largeur de la matrice et `blocksize` est la taille du bloc. Par exemple, `n` = 10000 et `blocksize` = 33.

Si votre implémentation de `transpose_blocking()` est correcte, la méthode de découpage en blocs devrait montrer une amélioration substantielle des performances par rapport à la version 'naïve'.

**Conseils :** (si vous ne savez pas par où commencer !)

Commencez par examiner la fonction `transpose_naive()` incluse dans le fichier. Notez que l'indice `y` parcourt verticalement TOUTE la matrice `src` dans une itération de la boucle externe avant de se remettre à `0`. Une autre façon de dire cela est que l'indice `x` est mis à jour seulement après que l'indice `y` ait parcouru toute la plage d'indices `[0 .. n-1]`. C'est le comportement que nous voudrions changer, on aimerait éviter de parcourir tous les indices du tableau.

En bref : remplissez `dst` avec un bloc carré à la fois, où chaque bloc est de dimension `blocsize` par `blocsize`.

Au lieu de mettre à jour `x` uniquement lorsque `y` ait parcouru tous les indices `0` à `n-1`, vous voudriez passer à la ligne suivante de `dst` après avoir parcouru la largeur (axe horizontal) d'un seul bloc. De même, vous voudriez parcourir seulement la hauteur d'un bloc (axe vertical) avant de passer au bloc suivant. Quelle est la taille d'un bloc ? Elle est donnée par le paramètre `blocksize` !

**Indication :** Une solution simple nécessite quatre boucles `for`.

Enfin, comme la largeur de la matrice `n` n'est pas nécessairement un multiple de la taille de bloc `blocksize`, la colonne et ligne finales de blocs seront tronquées (voir les blocs $$A_{3\_}$$ et $$A_{\_3}$$ dans la figure ci-dessous). Pour gérer cette situation, vous pouvez faire l'exercice en supposant au début que `n` est un multiple de `blocksize`, puis ajouter une condition quelque part dans le code pour ne rien faire lorsque vos index dépassent les limites de la matrice.

![CutBlocTransposition]({{site.baseurl}}/static_files/images/size_mismatch_matrix_transpose.png){: .aligncenter width="50%" height="50%"}     

Une fois que votre implémentation fonctionne corrèctement, l'étape suivante est d'effectuer une analyse des performances du programme.

## Modifier les dimensions des matrices

Exécutez votre code plusieurs fois avec une valeur de `blocksize` fixée à 20 et les valeurs 100, 1000, 2000, 5000 et 10000 pour `n`.

- À quel moment la version de transposition par << cache-blocking >> devient plus rapide que la version 'naïve' ?

- Pourquoi la version en << cache-blocking >> nécessite-t-elle que la matrice ait une certaine taille avant de surpasser les performances de la version 'naïve' ?

## Modifier la taille du bloc

Fixez `n` à 10000 et exécutez votre code avec une taille de bloc `blocksize` égale à 50, 100, 500, 1000, 5000.

- Comment les performances changent-elles lorsque la taille du bloc augmente ? Pourquoi ?


**Note finale :** Dans ces deux exercices, les caractéristiques associées aux caches de notre machine nous sont inconnus (c.-à-d. taille, structure, ...). Nous nous sommes simplement assurés que notre code expose un degré plus élevé de localité, et cela <!--, comme par magie,--> a amélioré considérablement les performances ! Cela indique que les caches, quels que soient leurs caractéristiques spécifiques, fonctionneront toujours mieux sur du code qui présente un bon niveau de localité (spatiale et/ou temporelle).


<!--
Les caches exploitent la localité spatiale et temporelle des accès mémoire. Bien que le code présente naturellement ces deux choses, nous pouvons généralement le modifier pour qu'il présente PLUS de ces deux choses, améliorant ainsi notre taux de réussite du cache et augmentant ainsi notre vitesse d'exécution.
-->
