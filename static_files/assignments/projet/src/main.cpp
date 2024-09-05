#include <getopt.h>

#include "fpng.h"
#include "mnt.h"

#define FICHIER_IMG_ACCUMULATIONS "accumulations.png"
#define FICHIER_IMG_DIRECTIONS    "directions.png" 
#define FICHIER_IMG_BASSINS       "bassins.png"

// Les données disponibles sont les suivantes

// #define filename "data/petit.txt"                  // ce fichier correspond à l'exemple de la fiche ipynb 6x6
// #define filename "data/jeu_essai.txt"           // ce fichier correspond à un extrait 32 x 32 de la dalle grd
//#define filename "data/grd_618360_6754408.txt"  // ce fichier correspond à la dalle grd complète 1025x1025
#define filename "data/alpes.txt"               // ce fichier correspond aux données sur les alpes 1024x1024

void usage(const char *progname)
{
    printf("Usage: %s [options]  -t <terrain_mnt.txt>  \n", progname);
    printf("Options du programme :\n");
    printf("  -t  --terrain        <terrain_mnt.txt>        fichier texte contenant la description du MNT\n");
    printf("  -d  --directions     <direction_mnt.png>      fichier de sortie pour stocker l'image des directions du flot, (%s par défaut)\n", FICHIER_IMG_DIRECTIONS);
    printf("  -a  --accumulations  <accumulation_mnt.png>   fichier de sortie pour stocker l'image des accumulations du flot, (%s par défaut)\n", FICHIER_IMG_ACCUMULATIONS);
    printf("  -b  --bassins        <bassin_mnt.png>         fichier de sortie pour stocker l'image des bassins ..., (%s par défaut)\n", FICHIER_IMG_BASSINS);
    printf("  -?  --help           Ce message\n");
}

int main(int argc, char** argv)
{
  fpng::fpng_init();

    int opt;
    static struct option long_options[] = {
        {"terrain", required_argument, NULL, 't'},
        {"directions", optional_argument, NULL, 'd'},
        {"accumulations", optional_argument, NULL, 'a'},
        {"bassins", optional_argument, NULL, 'b'},        
        {"help", no_argument, NULL, '?'},
        {NULL, 0, NULL, '\0'}};

    //  

    std::string terrain_filename;
    std::string direction_filename(FICHIER_IMG_DIRECTIONS);
    std::string accumulations_filename(FICHIER_IMG_ACCUMULATIONS);
    std::string bassins_filename(FICHIER_IMG_BASSINS);


    if (argc == 1)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    else
        while ((opt = getopt_long(argc, argv, "t:d:a:b:?", long_options, NULL)) != -1)
        {
            switch (opt)
            {
            case 't':
                terrain_filename = std::string(optarg);
                break;

            case 'd':
                direction_filename = std::string(optarg);
                break;

            case 'a':
                accumulations_filename = std::string(optarg);
                break;

            case 'b':
                bassins_filename = std::string(optarg);
                break;

            case '?':
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
    // end parsing of commandline options


  MNT mnt;
  mnt.lecture_mnt(terrain_filename.c_str());
 
  mnt.calcul_direction();
  std::vector<int>  pImage = mnt.direction_color_coding();
  fpng::fpng_encode_image_to_file(direction_filename.c_str(), pImage.data(), mnt.nb_lignes, mnt.nb_cols, 4);

  mnt.calcul_accumulation();
  std::vector<int>  pImage_acc = mnt.accumulation_color_coding();
  fpng::fpng_encode_image_to_file(accumulations_filename.c_str(), pImage_acc.data(), mnt.nb_lignes, mnt.nb_cols, 4);
 
  mnt.calcul_bassin();
  std::vector<int>  pImage_bassin = mnt.bassin_color_coding();
  fpng::fpng_encode_image_to_file(bassins_filename.c_str(), pImage_bassin.data(), mnt.nb_lignes, mnt.nb_cols, 4);

  return 0;
}


