import sys
import scanpy as sc
import pandas as pd

combination = sys.argv[1]
if combination not in ["I", "U"]:
    raise Exception("Invalid Combination Method")

pro3m = pd.read_csv('./data/alzPro/peaks_proteins_lf_3m.csv')
pro6m = pd.read_csv('./data/alzPro/peaks_proteins_lf_6m.csv')
pro9m = pd.read_csv('./data/alzPro/peaks_proteins_lf_9m.csv')

# Remove unnecessary columns
pro3m_v2 = pro3m[[col for col in pro3m.columns if col.startswith("Sample") and len(col) == 11 or col.startswith("Accession")]]
pro3m_v2 = pro3m_v2.reindex(sorted(pro3m_v2.columns), axis=1)

pro6m_v2 = pro6m[[col for col in pro6m.columns if col.startswith("Sample") and len(col) == 11 or col.startswith("Accession")]]
pro6m_v2 = pro6m_v2.reindex(sorted(pro6m_v2.columns), axis=1)

pro9m_v2 = pro9m[[col for col in pro9m.columns if col.startswith("Sample") and len(col) == 21 or col.startswith("Accession")]]
pro9m_v2.rename(columns={'Accession': 'Accession Intensity'}, inplace=True)
pro9m_v2.rename(columns=lambda x: x[:-10], inplace=True)
pro9m_v2 = pro9m_v2.reindex(sorted(pro9m_v2.columns), axis=1)

# Data Combination
if combination == "I":
    proteins_list = pro3m_v2.loc[pro3m_v2["Accession"].isin(pro6m_v2["Accession"])]
    proteins_list = proteins_list.loc[proteins_list["Accession"].isin(pro9m_v2["Accession"])]["Accession"]

    pro3m_v2 = pro3m_v2.loc[pro3m_v2["Accession"].isin(proteins_list)]
    pro6m_v2 = pro6m_v2.loc[pro6m_v2["Accession"].isin(proteins_list)]
    pro9m_v2 = pro9m_v2.loc[pro9m_v2["Accession"].isin(proteins_list)]
else:
    proteins_list = set(list(pro3m_v2["Accession"]) + list(pro6m_v2["Accession"]) + list(pro9m_v2["Accession"]))

    missing_3m = proteins_list - set(pro3m_v2["Accession"])
    for protein in missing_3m:
        pro3m_v2.loc[len(pro3m_v2)] = [protein] + [0]*16

    missing_6m = proteins_list - set(pro6m_v2["Accession"])
    for protein in missing_6m:
        pro6m_v2.loc[len(pro6m_v2)] = [protein] + [0]*16

    missing_9m = proteins_list - set(pro9m_v2["Accession"])
    for protein in missing_9m:
        pro9m_v2.loc[len(pro9m_v2)] = [protein] + [0]*16

    proteins_list = pro3m_v2.loc[pro3m_v2["Accession"].isin(pro6m_v2["Accession"])]
    proteins_list = proteins_list.loc[proteins_list["Accession"].isin(pro9m_v2["Accession"])]["Accession"]

# Merging DataFrames
pro3m_v2.sort_values(by=["Accession"], inplace=True)
pro6m_v2.sort_values(by=["Accession"], inplace=True)
pro9m_v2.sort_values(by=["Accession"], inplace=True)

pro3m_v2.reset_index(drop=True, inplace=True)
pro6m_v2.reset_index(drop=True, inplace=True)
pro9m_v2.reset_index(drop=True, inplace=True)

proteins_v2 = pd.concat([pro3m_v2, pro6m_v2.drop(columns=["Accession"], axis=1), pro9m_v2.drop(columns=["Accession"], axis=1)], axis=1)

# Cleaning and Saving
proteins_v2["Accession"] = proteins_v2["Accession"].apply(lambda x: x[:-6])
proteins_v2.index = proteins_v2["Accession"]
proteins_v2.drop(columns=["Accession"], inplace=True)

proObs = pd.read_excel("./data/alzPro/alz_sample_id.xlsx")

t1 = proteins_v2.T
t1.reset_index(drop=True, inplace=True)
t2 = proObs
t3 = pd.DataFrame(index=proteins_v2.index)

adata = sc.AnnData(X=t1, obs=t2, var=t3)

adata.write_h5ad(f"./data/alzPro_count_{combination}.h5ad")
