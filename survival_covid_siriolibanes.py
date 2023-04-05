import warnings
warnings.filterwarnings("ignore")

from utils import *

def main(args):

    # Leitura dos dados
    desfechos = pd.read_csv(os.path.join(args.dataset_dir, 'HSL_Desfechos_3.csv'), sep='|')
    exames = pd.read_csv(os.path.join(args.dataset_dir, 'HSL_Exames_3.csv'), sep='|')
    pacientes = pd.read_csv(os.path.join(args.dataset_dir, 'HSL_Pacientes_3.csv'), sep='|')
    df = preprocessing(args, desfechos, exames, pacientes)

    print("\n=> SURVIVAL ANALYSIS")
    print(df.head())
    data = get_columns_as_struct_array(args, df)
    results = surv_kaplan_meier_estimator(args, df, data, ['IC_SEXO', 'IDADE', 'IDADE_GRUPO'])
    print("\n [*] RESULTADOS KAPLAN-MEIER ESTIMATOR:")
    results_compare = surv_compare_survival(df, data, ['IC_SEXO', 'IDADE', 'IDADE_GRUPO'])
    print("\n [*] RESULTADOS LOG-RANK TESTS:")
    print('* IC_SEXO')
    print(results_compare['IC_SEXO']['pval'])
    print('* IDADE')
    print(results_compare['IDADE']['pval'])
    print('* IDADE_GRUPO')
    print(results_compare['IDADE_GRUPO']['pval']) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Survival Analysis Covid HSL')
    parser.add_argument('-d', "--dataset_dir", dest="dataset", type=str, help="Pasta contendo os dados", required=False, default='./HSL_Janeiro2021')
    parser.add_argument('-o', "--output_dir", dest="output_dir", type=str, help="Pasta de saida", required=False, default=os.getcwd())
    parser.add_argument('-e', "--evento", dest="evento", type=str, help="Evento avaliado", required=False, default='MELHORA')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
