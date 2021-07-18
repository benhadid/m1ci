---
type: assignment
date: 2021-06-02T08:00:00+1:00
title: 'Evaluation en ligne'
attachment: /static_files/assignments/devoir.zip
hide_from_announcements: True
due_event:
    type: due
    date: 2021-07-31T18:00:00+1:00
    description: 'Evaluation en ligne - à remettre'
---

# Comment soumettre votre devoir et comment il sera évalué

Ce devoir est à réaliser soit en groupe de deux ou individuellement (groupe de trois et plus non toléré). Voici la liste des fichiers à soumettre : 

```bash
 .
 ├── rot.c
 ├── mandelbrot.c
 ├── matmul16.c
 ├── matmul16.cl
 ├── matmul.c
 └── matmul.cl
```

 Créer un fichier `.zip` contenant les fichiers sus-mentionés, puis ajouter ce fichier en tant que pièce-jointe à ce [document](https://forms.gle/7LuZmT2i5UkoxX6Q8). 
 
 1. Tout devoir soumis après 18h00 31/07 sera **sanctionné de 5% de pénalité par minute de retard**.
 2. Une **seule tentative** vous sera accordée, vous ne pourrez plus modifier vos réponses après avoir cliqué sur 'Submit/Soumettre'. Assurez-vous donc d'avoir fini avant d'envoyer vos réponses. 
 3. Un seul envoi par binôme est nécessaire/accepté.
 4. N'oubliez pas de renseigner correctement votre Nom/prénom (et le nom/prénom de votre binôme le cas échéant). 
 5. **AUCUN** devoir ne sera accepté/évalué si envoyer par mail ! Veuillez utiliser le lien de Google Forms fourni ci-dessus.


## Méthode d'évaluation :

 1. Le programme ne compile même pas :  0%
 2. Le programme compile mais ne s'exécute pas correctement (segfault) :  0%
 3. Le programme s'exécute mais ne se termine pas (boucle infiniment)) :  0%
 4. Le programme s'exécute mais ne donne pas le résultat correct : jusqu'à 20% (dépend de ce que vous avez écrit)
 5. Le programme s'exécute, donne le résultat correct mais il est moins rapide que la version séquentielle : jusqu'à 30%
 6. Le programme s'exécute, donne le résultat correct mais prend autant de temps que la version séquentielle : 40%
 7. Le programme s'exécute, donne le résultat correct et il est plus rapide que la version séquentielle : 100%
 


# Exercice 1 (8 points)

La fonction ci-dessous fait pivoter un tableau `arr` à gauche : le code déplace tous les éléments du tableau vers la gauche et insert le premier élément à la fin du tableau.

```bash
void rot(int N, double* arr) {
  for (int i = 0; i < N - 1; i++) {
    double tmp = arr[i];
    arr[i] = arr[i+1];
    arr[i+1] = tmp;
  }
}
```

 1. Parallélisez ce code dans `rot.c` en utilisant OpenMP ( **4 points** ).

 2. Sans utiliser un tableau temporaire. (**4 points**)


# Exercice 2 (8 points)

L'[ensemble de Mandelbrot](https://fr.wikipedia.org/wiki/Ensemble_de_Mandelbrot) est une fractale qui peut être calculée par itération sur des points du plan complexe et en vérifiant si la séquence

$$
\left\{
      \begin{array}{ll}
          z_{0} = 0  \\
          z_{n} = z^{2}_{n} + c
      \end{array}
\right.
$$

est bornée.


Le programme fourni dans `mandelbrot.c` calcule la fractale de Mandelbrot en utilisant OpenMP. Le code, cependant, contient au moins 4 bugs.

 1. identifier et corriger ces bugs afin que le programme fonctionne correctement (**4 points**). 

 2. insérez des commentaires dans `mandlebrot.c` pour indiquer les bugs que vous avez trouvés et expliquez quels problèmes ils causaient et comment vous les avez résolus. (**4 points**).


# Exercice 3 (4 points)

Complétez le code dans `matmul16.cl` afin de réaliser une multiplication de matrices carrées $$(N \times N)$$ (A[N][K] * B[K][N] = C[N][N]). 
Opérez les changements nécessaires dans ce code et créer une nouvelle version dans `matmul.cl` pour réaliser des multiplications de matrices pas nécessairement carrées : (A[M][K] * B[K][N] = C[M][N]) avec M ≠ N. Apportez les changements nécessaire dans `matmul.c` pour cette deuxième version. 

Les topologies des threads dans un work-group (c.-à-d. nombre et organisation des threads dans un work-group) restent unchangés dans les deux versions (c.-à-d. workgroup de $16 \times 16$).

 1. Dans la version (fichiers `matmul16.c` et kernel `matmul16.cl`), vous pouvez supposer que M et N sont des multiples de 16 (**2 points**).

 2. Dans la deuxième version (fichiers `matmul.c` et kernel `matmul.cl`), votre programme doit réaliser des multiplications de matrices de tailles qui ne sont pas nécessairement des multiples de 16 (c.-à-d. dimensions M et N quelconques) (**2 points**).
