#ifndef MNT_H
#define MNT_H

#include <cstdint>
#include <cstdlib>

#include <vector>
#include <string>

struct MNT {

  int nb_lignes;
  int nb_cols;
  float no_value;
  int no_dir_value;
  int no_bassin_value;
  int num;
  int cst;


// Tableaux prédéfinis pour tester si un voisinage se déverse dans
// une cellule. Trois cas pour gérer le cas général et les 2 bords (droit et gauche)
  int dir_bord1[5]={5,6,7,8,1};
  int dir_bord2[5]={1,2,3,4,5};
  int dir_general[8]={5,6,7,8,1,2,3,4};

  std::vector<float> terrain;
  std::vector<int> direction;
  std::vector<int> accumulation;
  std::vector<int> bassin;

  MNT() : nb_lignes(0), nb_cols(0), no_value(__FLT_MAX__), no_dir_value(-9) , no_bassin_value(-10), num(100), cst(10000) {}
  

  void lecture_mnt(std::string nom);

   void calcul_direction();
   void calcul_accumulation();
   unsigned f_accumulation();

   void calcul_bassin() ;
   int f_bassin(int i, int j) ;

   std::vector<int> direction_color_coding();
   std::vector<int> accumulation_color_coding();
   std::vector<int> bassin_color_coding();
};

#endif // MNT_H
