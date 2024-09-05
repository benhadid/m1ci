#include <cstdio>
#include <cmath>

#include "mnt.h"

typedef union
{
  uint32_t flat;

  struct
  {
    uint8_t r, g, b, a;
  };

} color_t;

uint32_t rgba(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a)
{
  color_t color;
  color.r = _r;
  color.g = _g;
  color.b = _b;
  color.a = _a;

  return color.flat;
}

uint32_t conversion(int val)
{
  uint32_t color = rgba(255, 255, 255, 255);
  switch (val)
  {
  case 0:
    color = rgba(255, 255, 255, 255);
    break;
  case 1:
    color = rgba(255, 0, 0, 255);
    break;
  case 2:
    color = rgba(255, 127, 0, 255);
    break;
  case 3:
    color = rgba(255, 255, 0, 255);
    break;
  case 4:
    color = rgba(127, 255, 0, 255);
    break;
  case 5:
    color = rgba(0, 127, 0, 255);
    break;
  case 6:
    color = rgba(0, 0, 255, 255);
    break;
  case 7:
    color = rgba(0, 255, 255, 255);
    break;
  case 8:
    color = rgba(0, 0, 0, 255);
    break;
  default:
    color = rgba(255, 255, 255, 255);
  }
  return color;
}

// la fonction conversion_acc est une proposition pour convertir les flots d'accumulation
// en une couleur
uint32_t conversion_acc(int val, int nb_cols, int nb_lignes, int cst)
{

  uint32_t color;
  if (val == -1)
    color = rgba(0, 0, 0, 255);
  else
  {
    float test = cst * (val * 255.0) / (nb_cols * nb_lignes);
    if (test > 255)
      color = rgba(255, 0, 0, 255);
    else
      color = rgba(static_cast<uint8_t>(round(test)), 0, 0, 255);
  }
  return color;
}

int conversion_bassin(int val)
{
  uint32_t couleur[] = {rgba(0, 127, 0, 255),
                        rgba(255, 0, 0, 255),
                        rgba(127, 255, 0, 255),
                        rgba(255, 127, 0, 255),
                        rgba(127, 255, 127, 255),
                        rgba(0, 0, 255, 255),
                        rgba(0, 255, 127, 255),
                        rgba(255, 255, 255, 255),
                        rgba(127, 127, 127, 255),
                        rgba(0, 255, 0, 255)};
  int v = val % 10;

  if (val == -10)
    return rgba(0, 0, 0, 255);
  else
    switch (v)
    {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
      return couleur[v];
    default:
      return rgba(50, 50, 50, 255);
    }
}

// Cette fonction permet juste de convertir le code des directions par des couleurs.

std::vector<int> MNT::direction_color_coding()
{
  std::vector<int> cur_img(nb_lignes * nb_cols);

  for (int i = 0; i < nb_lignes; i++)
    for (int j = 0; j < nb_cols; j++)
      cur_img[i * nb_cols + j] = conversion(direction[(i + 1) * nb_cols + j]);

  return cur_img;
}

std::vector<int> MNT::accumulation_color_coding()
{
  std::vector<int> cur_img(nb_lignes * nb_cols);

  for (int i = 0; i < nb_lignes; i++)
    for (int j = 0; j < nb_cols; j++)
    {
      cur_img[i * nb_cols + j] = conversion_acc(accumulation[(i + 1) * nb_cols + j], nb_cols, nb_lignes, cst);
    }
  return cur_img;
}

std::vector<int> MNT::bassin_color_coding()
{
  std::vector<int> cur_img(nb_lignes * nb_cols);

  for (int i = 0; i < nb_lignes; i++)
    for (int j = 0; j < nb_cols; j++)
    {
      cur_img[i * nb_cols + j] = conversion_bassin(bassin[i * nb_cols + j]);
    }
  return cur_img;
}

// Cette fonction permet de lire le fichier, d'initialiser nb_lignes, nb_cols et de construire terrain.
void MNT::lecture_mnt(std::string nom)
{
  FILE *f = fopen(nom.c_str(), "r");
  if (f != NULL)
  {
    int tmp;
    if (fscanf(f, "%d", &tmp) == 1)
      ;
    nb_lignes = tmp;
    if (fscanf(f, "%d", &tmp) == 1)
      ;
    nb_cols = tmp;
    if (fscanf(f, "%d", &tmp) == 1)
      ;
    if (fscanf(f, "%d", &tmp) == 1)
      ;
    if (fscanf(f, "%d", &tmp) == 1)
      ;
    if (fscanf(f, "%f", &no_value) == 1)
      ;

    terrain = std::vector<float>((2 + nb_lignes) * nb_cols, 0.0);

    for (int i = 0; i < nb_lignes; i++)
      for (int j = 0; j < nb_cols; j++)
        if (fscanf(f, "%f", &(terrain[(i + 1) * nb_cols + j])) == 1)
          ;
  }

  for (int j = 0; j < nb_cols; j++)
  {
    // première et dernière lignes sont initialisées à no_value
    terrain[j] = no_value;
    terrain[(nb_lignes + 1) * nb_cols + j] = no_value;
  }
}

// La fonction pour calculer les directions est déjà implémentée
// Elle prend en paramètre un terrain (paramètre data) et elle renvoie les directions dans le tableau dir.
// Attention data est de taille (n_l+2) x n_c afin de prendre en compte les ghosts.
// Ce n'est pas nécessaire en séquentiel mais cette fonction sera disponible également pour votre parallélisation.
// dir est de taille n_l x n_c
int f_bord1(float ref, float *tab, float no_value)
{
  float min = ref;
  int code = 0;
  for (int i = 0; i < 5; i++)
    if (tab[i] != no_value && tab[i] < min)
    {
      min = tab[i];
      code = i + 1;
    }
  return code;
}

int f_bord2(float ref, float *tab, float no_value)
{
  float min = ref;
  int code = -1;
  for (int i = 0; i < 5; i++)
    if (tab[i] != no_value && tab[i] < min)
    {
      min = tab[i];
      code = i;
    }

  switch (code)
  {
  case -1:
    return 0;
  case 4:
    return 1;
  default:
    return code + 5;
  }
}

int f(float ref, float *tab, float no_value)
{

  float min = ref;
  int code = 0;
  for (int i = 0; i < 8; i++)
    if (tab[i] != no_value && tab[i] < min)
    {
      min = tab[i];
      code = i + 1;
    }
  return code;
}

void MNT::calcul_direction()
{
  int x, y;
  int x1, y1;
  int x2, y2;
  float tab[8];
  float tab_bord[5];

  direction = std::vector<int>((nb_lignes + 2) * nb_cols);

  for (int i = 0; i < nb_lignes; i++)
  {
    x = i + 1;
    x1 = x - 1;
    x2 = x + 1;
    for (int j = 0; j < nb_cols; j++)
    {
      y = j;
      y1 = y - 1;
      y2 = y + 1;
      float val = terrain[x * nb_cols + y];
      if (val != no_value)
      {
        if (j == 0)
        { // calcul de direction sur le bord gauche
          tab_bord[0] = terrain[x1 * nb_cols + y];
          tab_bord[1] = terrain[x1 * nb_cols + y2];
          tab_bord[2] = terrain[x * nb_cols + y2];
          tab_bord[3] = terrain[x2 * nb_cols + y2];
          tab_bord[4] = terrain[x2 * nb_cols + y];
          direction[i * nb_cols + j + nb_cols] = f_bord1(val, tab_bord, no_value);
        }
        else if (j == (nb_cols - 1))
        { // calcul de direction sur le bord droit
          tab_bord[0] = terrain[x2 * nb_cols + y];
          tab_bord[1] = terrain[x2 * nb_cols + y1];
          tab_bord[2] = terrain[x * nb_cols + y1];
          tab_bord[3] = terrain[x1 * nb_cols + y1];
          tab_bord[4] = terrain[x1 * nb_cols + y];
          direction[i * nb_cols + j + nb_cols] = f_bord2(val, tab_bord, no_value);
        }
        else
        { // calcul de direction à l'interieur du terrain
          tab[0] = terrain[x1 * nb_cols + y];
          tab[1] = terrain[x1 * nb_cols + y2];
          tab[2] = terrain[x * nb_cols + y2];
          tab[3] = terrain[x2 * nb_cols + y2];
          tab[4] = terrain[x2 * nb_cols + y];
          tab[5] = terrain[x2 * nb_cols + y1];
          tab[6] = terrain[x * nb_cols + y1];
          tab[7] = terrain[x1 * nb_cols + y1];
          direction[i * nb_cols + j + nb_cols] = f(val, tab, no_value);
        }
      }
      else
      {
        direction[i * nb_cols + j + nb_cols] = no_dir_value;
      }
    }
  }
}

void MNT::calcul_accumulation()
{
  // il y a deux lignes supplémentaires dans direction
  // la première est mise à 1 pour indiquer que la cellule correspondante
  // se déverse vers le haut
  // la dernière est mise à 5 pour indiquer que la cellule correspondante
  // se déverse vers le bas
  for (int i = 0; i < nb_cols; i++)
  {
    direction[i] = 1;
    direction[(nb_lignes + 1) * nb_cols + i] = 5;
  }

  accumulation = std::vector<int>((nb_lignes + 2) * nb_cols, -1);

  // Pour les flots d'accumulations les 2 lignes supplémentaires sont
  // mises à 0. Mais via la convention de la matrice direction
  // on n'utilisera pas ces lignes dans le calcul.
  for (int i = 0; i < nb_cols; i++)
  {
    accumulation[i] = 0;
    accumulation[(nb_lignes + 1) * nb_cols + i] = 0;
  }

  int stop = 0;
  while (stop != 1)
  {
    stop = f_accumulation();
  }
}

int f_acc(int *tab, int no_value, int *dir_bord, int l)
{
  int val = 0;
  int nb1 = 0;
  int nb2 = 0;

  for (int i = 0; i < l; i++)
    if (tab[l + i] == dir_bord[i])
    {
      nb1++;
      if (tab[i] != -1)
      {
        val += tab[i];
        nb2++;
      }
    }

  if (nb1 == nb2)
    val++;
  else
    val = -1;
  return val;
}

unsigned MNT::f_accumulation()
{
  unsigned stop = 0;
  int nb_non_calculs = 0;
  int x, y;
  int x1, y1;
  int x2, y2;
  int tab[16];
  int tab_bord[10];
  for (int i = 0; i < nb_lignes; i++)
  {
    x = i + 1;
    x1 = x - 1;
    x2 = x + 1;
    for (int j = 0; j < nb_cols; j++)
    {
      y = j;
      y1 = y - 1;
      y2 = y + 1;
      int d = direction[x * nb_cols + y];
      if (d != no_dir_value)
      {
        if (accumulation[x * nb_cols + j] == -1)
        {
          if (j == 0)
          {
            tab_bord[0] = accumulation[x1 * nb_cols + y];
            tab_bord[1] = accumulation[x1 * nb_cols + y2];
            tab_bord[2] = accumulation[x * nb_cols + y2];
            tab_bord[3] = accumulation[x2 * nb_cols + y2];
            tab_bord[4] = accumulation[x2 * nb_cols + y];
            tab_bord[5] = direction[x1 * nb_cols + y];
            tab_bord[6] = direction[x1 * nb_cols + y2];
            tab_bord[7] = direction[x * nb_cols + y2];
            tab_bord[8] = direction[x2 * nb_cols + y2];
            tab_bord[9] = direction[x2 * nb_cols + y];
            int res = f_acc(tab_bord, no_dir_value, dir_bord1, 5);
            if (res != -1)
              accumulation[x * nb_cols + j] = res;
            else
            {
              nb_non_calculs++;
            }
          }
          else if (j == (nb_cols - 1))
          {
            tab_bord[0] = accumulation[x2 * nb_cols + y];
            tab_bord[1] = accumulation[x2 * nb_cols + y1];
            tab_bord[2] = accumulation[x * nb_cols + y1];
            tab_bord[3] = accumulation[x1 * nb_cols + y1];
            tab_bord[4] = accumulation[x1 * nb_cols + y];
            tab_bord[5] = direction[x2 * nb_cols + y];
            tab_bord[6] = direction[x2 * nb_cols + y1];
            tab_bord[7] = direction[x * nb_cols + y1];
            tab_bord[8] = direction[x1 * nb_cols + y1];
            tab_bord[9] = direction[x1 * nb_cols + y];
            int res = f_acc(tab_bord, no_dir_value, dir_bord2, 5);
            if (res != -1)
              accumulation[x * nb_cols + j] = res;
            else
            {
              nb_non_calculs++;
            }
          }
          else
          {
            tab[0] = accumulation[x1 * nb_cols + y];
            tab[1] = accumulation[x1 * nb_cols + y2];
            tab[2] = accumulation[x * nb_cols + y2];
            tab[3] = accumulation[x2 * nb_cols + y2];
            tab[4] = accumulation[x2 * nb_cols + y];
            tab[5] = accumulation[x2 * nb_cols + y1];
            tab[6] = accumulation[x * nb_cols + y1];
            tab[7] = accumulation[x1 * nb_cols + y1];
            tab[8] = direction[x1 * nb_cols + y];
            tab[9] = direction[x1 * nb_cols + y2];
            tab[10] = direction[x * nb_cols + y2];
            tab[11] = direction[x2 * nb_cols + y2];
            tab[12] = direction[x2 * nb_cols + y];
            tab[13] = direction[x2 * nb_cols + y1];
            tab[14] = direction[x * nb_cols + y1];
            tab[15] = direction[x1 * nb_cols + y1];
            int res = f_acc(tab, no_dir_value, dir_general, 8);
            if (res != -1)
              accumulation[x * nb_cols + j] = res;
            else
              nb_non_calculs++;
          }
        }
      }
    }
  }
  if (nb_non_calculs == 0)
    stop = 1;
  return stop;
}

void MNT::calcul_bassin()
{

  bassin = std::vector<int>(nb_lignes * nb_cols, -1);

  for (int i = 0; i < nb_lignes; i++)
    for (int j = 0; j < nb_cols; j++)
    {
      if (direction[i * nb_cols + j + nb_cols] != no_dir_value)
      {
        if (bassin[i * nb_cols + j] == -1)
          bassin[i * nb_cols + j] = f_bassin(i, j);
      }
      else
      {
        bassin[i * nb_cols + j] = no_bassin_value;
      }
    }
}

int MNT::f_bassin(int i, int j)
{
  /* check if i,j belongs to this process... if not pass on the coordinates to the right process and wait for the return value */


  if (direction[i * nb_cols + j + nb_cols] == 0)
  {
    /* set bassin(i,j) if necessary and return */

    if (bassin[i * nb_cols + j] == -1)
    {
      num++;
      bassin[i * nb_cols + j] = num;
    }

    return bassin[i * nb_cols + j];
  }
  else
  {
    /* direction(i,j) != 0 */

    if (bassin[i * nb_cols + j] != -1)
      return bassin[i * nb_cols + j];
    else
    { /* bassin == -1 */
      int etiquette;
      switch (direction[i * nb_cols + j + nb_cols])
      {
      case 1:
        etiquette = f_bassin(i - 1, j);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 2:
        etiquette = f_bassin(i - 1, j + 1);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 3:
        etiquette = f_bassin(i - 1, j);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 4:
        etiquette = f_bassin(i + 1, j + 1);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 5:
        etiquette = f_bassin(i + 1, j);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 6:
        etiquette = f_bassin(i + 1, j - 1);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 7:
        etiquette = f_bassin(i, j - 1);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      case 8:
        etiquette = f_bassin(i - 1, j - 1);
        bassin[i * nb_cols + j] = etiquette;
        return etiquette;
      }
    }
  }
  return 0;
}
