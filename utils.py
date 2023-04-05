from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, List
from datetime import date
import pandas as pd
import numpy as np
# import unidecode
import argparse
# import seaborn
import joblib
import os

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.compare import compare_survival


only_covid = True


def get_days_from_date(date_string):
    """Transforma date_string DD/MM/AAAA na contagem de numero de dias corridos a partir de 01/11/2019."""
    if date_string == 'DDMMAA':
        return date_string
    else:
        f_date = date(2019, 11, 1)
        day, month, year = date_string.split('/')
        l_date = date(int(year), int(month), int(day))
        delta = l_date - f_date
        return delta.days


def preprocessing_pacientes(args, pacientes):
    print('\n\n=> PACIENTES')
    print('[*] Pacientes [entrada]:', len(pacientes.ID_PACIENTE.unique()))
    selected_columns = ['ID_PACIENTE', 'IC_SEXO', 'aa_nascimento']
    pacientes = pacientes[selected_columns]
    pacientes = pacientes.drop(pacientes[pacientes.aa_nascimento == 'YYYY'].index)
    # 'AAAA': menor ou igual a 1930
    # 'YYYY': quaisquer outros anos, anonimizado
    # considerando ano = 1930 para aa_nascimento <= 1930
    pacientes["aa_nascimento"] = pacientes.aa_nascimento.apply(lambda x: "1930" if x == 'AAAA' else x)
    pacientes["IDADE"] = pacientes.aa_nascimento.apply(lambda x: 2020 - int(x))
    pacientes["IDADE_GRUPO"] = pacientes.aa_nascimento.apply(get_age_group)
    pacientes.to_csv(args.output_dir + 'pacientes_out.csv')
    print('[*] Pacientes [saida]:', len(pacientes.ID_PACIENTE.unique()))
    return pacientes


def preprocessing_exames(args, exames, pacientes):
    print('\n\n=> EXAMES')
    print('[*] Exames [entrada]:', len(exames))
    selected_columns = ['ID_PACIENTE', 'ID_ATENDIMENTO', 'DT_COLETA', 'DE_EXAME', 'DE_ANALITO', 'DE_RESULTADO']
    exames = exames[selected_columns]
    exames = exames[exames.ID_PACIENTE.isin(pacientes.ID_PACIENTE.unique())]
    exames.DT_COLETA = exames.DT_COLETA.apply(get_days_from_date)
    exames['DE_EXAME'] = exames['DE_EXAME'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    exames['DE_ANALITO'] = exames['DE_ANALITO'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    exames['DE_RESULTADO'] = exames['DE_RESULTADO'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    exames = concat_exame_analito(exames)
    EXAMES_ANALITO_COVID_19 = get_list_of_exames_covid(exames)
    exames = pivoting_exames_analito(exames, EXAMES_ANALITO_COVID_19, only_covid)

    # filtrar: apenas os pacientes que testaram positivo para covid em algum momento
    # isso aqui so faz sentido quando only_covid = True
    if only_covid:
        exames = get_only_covid_patients(exames)

    exames.to_csv(args.output_dir + 'exames_out_only_covid_' + str(only_covid) + '.csv')
    print('[*] Exames [saida]:', len(exames))
    print('[*] Exames->Pacientes [saida]:', len(exames.ID_PACIENTE.unique()))

    return exames


def pivoting_exames_analito(exames, EXAMES_ANALITO_COVID_19, only_covid=True):

    if only_covid:

        print('\n[*] Considerando apenas exames de Covid-19 na análise.')
        # isso aqui so faz sentido se eu so tiver exame de covid:::
        exames = exames[exames.DE_EXAME_ANALITO.isin(EXAMES_ANALITO_COVID_19)]
        # padronizacao dos exames em DETECTADO (1) ou NAO DETECTADO (0)
        # esses valores foram verificados por inspecao da base de dados:
        positive_for_covid = ['DETECTADO', 'DETECTADO (POSITIVO)', 'DETECTAVEL', 'REAGENTE']
        exames["DE_RESULTADO_COVID"] = exames.DE_RESULTADO.apply(lambda x: 1 if x in positive_for_covid else 0)

    else:

        # criar coluna de resultado do covid 
        selected_exames = get_list_of_exames(exames)
        print('\n[*] Pares Exame-Analito gerais considerados na análise:', len(selected_exames))
        print(selected_exames)  
        selected_exames_analito = selected_exames + EXAMES_ANALITO_COVID_19
        exames = exames[exames.DE_EXAME_ANALITO.isin(selected_exames_analito)]

        # padronizacao dos exames em DETECTADO (1) ou NAO DETECTADO (0)
        # esses valores foram verificados por inspecao da base de dados:
        positive_for_covid = ['DETECTADO', 'DETECTADO (POSITIVO)', 'DETECTAVEL', 'REAGENTE']

        new_columns = exames.DE_EXAME_ANALITO.unique()
        exames[new_columns] = np.nan
        print("[*] Pivotando exames... Total:", len(exames))
        for index, row in exames.iterrows():
            print(index, len(exames), end='\r')
            exame_row = row["DE_EXAME_ANALITO"]
            resultado_row = row["DE_RESULTADO"]
            if exame_row in EXAMES_ANALITO_COVID_19:
                if resultado_row in positive_for_covid: 
                    exames.loc[index, "DE_RESULTADO_COVID"] = 1
                else:
                    exames.loc[index, "DE_RESULTADO_COVID"] = 0

            else:
                exames.loc[index, exame_row] = resultado_row
        print('\n')

    print(exames.columns)
    print(exames.head())

    return exames


def preprocessing_desfechos(args, desfechos, exames):
    print('\n\n=> DESFECHOS')
    selected_columns = ['ID_PACIENTE', 'ID_ATENDIMENTO', 'DT_ATENDIMENTO',
       'DE_TIPO_ATENDIMENTO', 'DT_DESFECHO', 'DE_DESFECHO']
    desfechos = desfechos[selected_columns]
    desfechos = desfechos[desfechos.ID_PACIENTE.isin(exames.ID_PACIENTE.unique())]
    desfechos.DT_ATENDIMENTO = desfechos.DT_ATENDIMENTO.apply(get_days_from_date)
    desfechos.DT_DESFECHO = desfechos.DT_DESFECHO.apply(get_days_from_date)
    desfechos['DE_DESFECHO'] = desfechos['DE_DESFECHO'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    desfechos["STATUS_OBITO"] = False
    desfechos["STATUS_MELHORA"] = False
    desfechos = fill_status_obito_e_melhora(desfechos)

    # censura dados de acordo com o evento
    if args.evento == 'OBITO':
        desfechos = desfechos[desfechos.STATUS_MELHORA == False] # dropo todos os pacientes que receberam alta
    if args.evento == 'MELHORA':
        desfechos = desfechos[desfechos.STATUS_OBITO == False]  # dropo todos os pacientes que vieram a obito

    desfechos.to_csv(args.output_dir + 'desfechos_out.csv')
    print('Desfechos:', (len(desfechos)))
    print('Desfechos - Pacientes:', (len(desfechos.ID_PACIENTE.unique())))
    desfechos.to_csv(args.output_dir + 'desfechos_out.csv')
    return desfechos


def concat_exame_analito(exames):
    exames["DE_EXAME_ANALITO"] = ""
    exames_de_exame = exames.DE_EXAME.values
    exames_de_analito = exames.DE_ANALITO.values
    pares = []
    for i in range(len(exames_de_exame)):
        exame = exames_de_exame[i]
        analito = exames_de_analito[i]
        pares.append(exame + '_' + analito)
    exames["DE_EXAME_ANALITO"] = pares
    return exames


def get_list_of_exames(exames):
    """Retorna a lista de exames a serem considerados na analise. Essas colunas serao pivotadas."""
    selected_exames = []
    exames_analito = exames.DE_EXAME_ANALITO.values
    selected_exames = ['Vitamina D', 'Acido Ascorbico', 'Filtracao Glomular', 'Proteina C Reativa',
                    'Ureia', 'Creatininina', 'Dimeros', 'Hemograma'] 

    exames_analitos_selected = []
    for keyword in selected_exames:
        find_exame = [s for s in exames_analito if keyword in s]
        find_exame = list(set(find_exame))
        exames_analitos_selected = exames_analitos_selected + find_exame

    return list(set(exames_analitos_selected))


def get_list_of_exames_covid(exames):

    pares = exames.DE_EXAME_ANALITO.values

    find_coronavirus = [s for s in pares if 'Coronavirus' in s]
    find_coronavirus = list(set(find_coronavirus))
    find_covid = [s for s in pares if 'Covid' in s]
    find_covid = list(set(find_covid))
    find_covid = find_covid + find_coronavirus
    exames_covid = list(set(find_covid))

    descarte = [s for s in exames_covid if 'Coronavirus OC43' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'Coronavirus NL63' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'Coronavirus 229E' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'Coronavirus HKU1' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'IgM' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'IgA' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'IgG' in s]
    exames_covid = list(set(exames_covid) - set(descarte))
    descarte = [s for s in exames_covid if 'IGG' in s]
    exames_covid = list(set(exames_covid) - set(descarte))

    exames_covid = ['Sars Cov-2, Teste Molecular Rapido Para Deteccao, Varios Materiais_Covid 19, Deteccao por PCR',
    'COVID-19-PCR para SARS-COV-2, Varios Materiais (Fleury)_Coronavirus (2019-nCoV)', 
    'Deteccao de Coronavirus (NCoV-2019) POR PCR (Anatomia Patologica)_Deteccao de Coronavirus (NCoV-2019) POR PCR (Anatomia Patologica)']
    # 'Sars Cov-2, Teste Molecular Rapido Para Deteccao, Varios Materiais_Covid 19, Material']

    print("\n[*] Pares Exame-Analito para Covid-19 considerados na análise:")
    print(exames_covid)

    return exames_covid


def fill_status_obito_e_melhora(desfechos):
    """Preenche coluna que indica pacientes que vieram a obito e que tiveram melhora. Mesmo que tenham recebido alta no meio da linha do tempo."""

    print('Preenchendo colunas de OBITO e MELHORA...')

    desfechos_alta = ['Alta a pedido', 'Alta medica curado', 'Alta medica melhorado',
    'Alta medica Inalterado', 'Alta Administrativa', 'Alta por abandono']

    # 'Transferência Inter-Hospitalar Externa - Serviço de Ambulância'
    # 'Transferência Inter-Hospitalar Externa - Transporte Próprio'
    # 'Óbito nas primeiras 48hs de internação sem necrópsia não agônico'
    # 'Óbito após 48hs de internação sem necrópsia'
    # 'Desistência do atendimento'
    # 'Óbito após 48hs de internação com necrópsia'
    # 'Óbito nas primeiras 48hs de internação sem necrópsia agônico']

    for index, row in desfechos.iterrows():
        paciente = row["ID_PACIENTE"]
        desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
        # print('\n\n')
        # print(desfechos_paciente)
        find_obito = [s for s in desfechos_paciente.DE_DESFECHO.values if 'Obito' in s]
        find_melhora = [s for s in desfechos_paciente.DE_DESFECHO.values if s in desfechos_alta]
        if find_obito != []:
            desfechos.loc[index, "STATUS_OBITO"] = True
        elif find_obito == [] and find_melhora != []:
            # STATUS_MELHORA: não veio a óbito a NENHUM momento.
            desfechos.loc[index, "STATUS_MELHORA"] = True

    return desfechos


def get_only_covid_patients(exames):
    """Filtra apenas os pacientes que EM ALGUM MOMENTO TESTARAM POSITIVO."""
    pacientes_covid = []
    for index, row in exames.iterrows():
        paciente = row["ID_PACIENTE"]
        exames_paciente = exames[exames.ID_PACIENTE == paciente]
        positive_date = get_positive_covid_date(exames_paciente)
        if positive_date is not None:
            pacientes_covid.append(paciente)
        else:
            continue

    return exames[exames.ID_PACIENTE.isin(pacientes_covid)]


def get_positive_covid_date(exames_paciente):
    """Retorna a PRIMEIRA DATA em que o paciente TESTOU POSITIVO para covid-19.
    Caso nao tenha TESTADO POSITIVO em NENHUM momento, retorna None"""
    exames_paciente_sorted = exames_paciente.sort_values(by='DT_COLETA')
    positive_date = None
    for index, row in exames_paciente_sorted.iterrows():
        if row["DE_RESULTADO_COVID"] == 1:
            positive_date = row["DT_COLETA"]
            return positive_date
        else:
            continue
    return positive_date


def get_status_obito(paciente, desfechos):
    """Retorna se o paciente veio a obito"""
    desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
    return desfechos_paciente["STATUS_OBITO"].values[0]


def get_status_melhora(paciente, desfechos):
    """Retorna se o paciente apresentou melhora"""
    desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
    return desfechos_paciente["STATUS_MELHORA"].values[0]


def get_status_alta(paciente, desfechos):
    """Retorna se o paciente recebeu alta"""
    desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
    return desfechos_paciente["STATUS_ALTA"].values[0]


def get_survival_obito(paciente, exames, desfechos, status_obito, last_date):

    exames_paciente = exames[exames.ID_PACIENTE == paciente]
    exames_paciente = exames_paciente.sort_values(by="DT_COLETA")
    desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
    desfechos_paciente = desfechos_paciente.sort_values(by="DT_ATENDIMENTO")

    if status_obito == 1: # se faleceu, pego o momento
        death_date = exames_paciente.DT_COLETA.values[-1]
    if status_obito == 0: # se nao faleceu, pegar a última data da base
        death_date = last_date 

    # checar isso
    positive_date = get_positive_covid_date(exames_paciente)
    # considero a data de admissao / data_atendimento na planilha de desfechos
    if positive_date != None:
        start_date = positive_date # desfechos_paciente.DT_ATENDIMENTO.values[0]
        survival_in_days = death_date - start_date
        return survival_in_days
    else:
        return -1


def get_survival_melhora(paciente, exames, desfechos, status_melhora, last_date):

    exames_paciente = exames[exames.ID_PACIENTE == paciente]
    exames_paciente = exames_paciente.sort_values(by="DT_COLETA")

    desfechos_paciente = desfechos[desfechos.ID_PACIENTE == paciente]
    desfechos_paciente = desfechos_paciente.sort_values(by="DT_DESFECHO")

    melhora_date = desfechos_paciente.DT_DESFECHO.values[-1] # ultimo desfecho de melhora -> curado

    positive_date = get_positive_covid_date(exames_paciente) # primeiro dia que testou positivo - 26/04/2020
    if positive_date is not None:
        start_date = positive_date
        survival_in_days = melhora_date - start_date
        return survival_in_days
    else:
        return -1


def preprocessing(args, desfechos, exames, pacientes):
    """Faz o pré-processamento dos dados"""

    print("PRE-PROCESSAMENTO DOS DADOS")
    print("EVENTO:", args.evento)

    pacientes = preprocessing_pacientes(args, pacientes)
    print(pacientes.head()) 
    exames = preprocessing_exames(args, exames, pacientes)
    last_date = np.max(exames.DT_COLETA)
    print(exames.head())    
    desfechos = preprocessing_desfechos(args, desfechos, exames)
    print(desfechos.head()) 

    print('\n\n=> FAZENDO O MERGE PARA UM UNICO DATAFRAME')

    if args.evento == 'OBITO':
        df = pacientes[pacientes.ID_PACIENTE.isin(desfechos.ID_PACIENTE.unique())].copy()
        print('Final dataframe:', len(df))
        df["DAYS_EVENT_OBITO"] = ""
        df["STATUS_MELHORA"] = ""
        print('Gerando variavel DAYS_EVENT_OBITO')

        for index, row in df.iterrows():
            paciente = row["ID_PACIENTE"]
            status_obito = get_status_obito(paciente, desfechos)
            survival_obito = get_survival_obito(paciente, exames, desfechos, status_obito, last_date)
            df.loc[index, "DAYS_EVENT_OBITO"] = survival_obito
            df.loc[index, "STATUS_OBITO"] = status_obito

        df.to_csv(args.output_dir + 'obito_df.csv')

    if args.evento == 'MELHORA':
        df = pacientes[pacientes.ID_PACIENTE.isin(desfechos.ID_PACIENTE.unique())].copy()
        print('Final dataframe:', len(df))
        print(len(df.ID_PACIENTE.unique()))
        df["DAYS_EVENT_MELHORA"] = ""
        df["STATUS_MELHORA"] = ""
        print('Gerando variavel DAYS_EVENT_MELHORA')

        # get survival_in_days
        PACIENTES_TO_DROP = []
        for index, row in df.iterrows():
            paciente = row["ID_PACIENTE"]
            # print('Paciente:', paciente)
            status_melhora = get_status_melhora(paciente, desfechos)
            survival_melhora = get_survival_melhora(paciente, exames, desfechos, status_melhora, last_date)
            if survival_melhora < 0:
                # há alguns pacientes com data de coleta após o período do mesmo no hospital.
                # esses serão desconsiderados na análise.
                PACIENTES_TO_DROP.append(paciente)
            df.loc[index, "DAYS_EVENT_MELHORA"] = survival_melhora
            df.loc[index, "STATUS_MELHORA"] = status_melhora

        df = df.drop(df[df['ID_PACIENTE'].isin(PACIENTES_TO_DROP)].index)
        df.to_csv(args.output_dir + 'melhora_df.csv')

    return df

def surv_kaplan_meier_estimator(args, df, data, groups=None):
    filename = args.evento
    columns = data.dtype.names
    print("[*] Gerando plots de Kaplan-Meier")

    results_dict = dict()
    time, survival_prob = kaplan_meier_estimator(data[columns[0]], data[columns[1]])
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    axs.step(time, survival_prob, where="post")
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    fig.savefig(args.output_dir + 'kaplan_meier_' + filename + '.png')
    plt.close()

    results_dict[filename] = {'time': time, 'survival_prob': survival_prob}
    
    label_for_plotting = {
        'IC_SEXO': 'Sexo',
        'IDADE': 'Idade',
        'IDADE_GRUPO': 'Idade (Grupo)'
    }

    if groups:
        for group in groups:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            time_ = []
            survival_prob_ = []
            for value in df[group].unique():

                mask = df[group] == value
                time, survival_prob = kaplan_meier_estimator(data[columns[0]][mask], data[columns[1]][mask])
                axs.step(time, survival_prob, where="post", label="%s = %s (n = %d)" % (label_for_plotting[group], value, mask.sum()))
                time_.append(time)
                survival_prob_.append(survival_prob)
            
            results_dict[group] = {'time': time_, 'survival_prob': survival_prob_}

            plt.ylabel("$\hat{S}(t)$", fontsize = 16)
            plt.xlabel("$t$", fontsize = 16)
            plt.legend(loc="upper right", fontsize = 16)
            fig.savefig(args.output_dir + 'kaplan_meier_' + filename + '_' + group + '.png')
            plt.close()
            

    return results_dict
    

def get_age_group(ano_de_nascimento):
    """Retorna o grupo referente a faixa etária do paciente"""

    ano = int(ano_de_nascimento)
    if ano > 2010:
        return 5
    elif ano > 1990:
        return 4
    elif ano > 1970:
        return 3
    elif ano > 1950:
        return 2
    elif ano > 1930:
        return 1
    else:
        return 0


def get_columns_as_struct_array(args, df):
    column_status = 'STATUS_' + args.evento
    column_days = 'DAYS_EVENT_' + args.evento
    df[column_status] = df[column_status].astype('bool')
    df[column_days] = df[column_days].astype('float64')
    return df[[column_status, column_days]].to_records(index=False)


def surv_compare_survival(df, data, groups):
    results = dict()
    for group in groups:
        group_indicator = df[group].values
        chisq, pval, table, covar = compare_survival(data, group_indicator, return_stats=True)
        results[group] = {'chisq': chisq, 'pval': pval, 'table': table, 'covar': covar}
    return results