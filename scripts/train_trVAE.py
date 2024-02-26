import sys
import numpy as np
import scanpy as sc

import reptrvae

normalize = False
data_name = sys.argv[1]
print(sys.argv)
# specific_cell_type = sys.argv[2]

if data_name == "alzPro":
    conditions = ["WT", "HET"]
    target_conditions = ["HET"]
    source_condition = "WT"
    target_condition = "HET"
    labelencoder = {"WT": 0, "HET": 1}
    cell_type_key = "Timepoint"
    condition_key = "Group"
    if len(sys.argv) >= 3:
        combination = sys.argv[2]
    else:
        combination = None
    if len(sys.argv) == 4:
        specific_celltype = sys.argv[3]
    else:
        specific_celltype = "all"
    dname = data_name

elif data_name == "alzPro-time":
    conditions = ["3m", "6m", "9m"]
    source_condition = "3"
    target_condition = "9m"
    target_conditions = [target_condition]
    labelencoder = {"3m": 0, "6m": 1, "9m": 2}
    cell_type_key = "Group"
    condition_key = "Timepoint"
    if len(sys.argv) >= 3:
        combination = sys.argv[2]
    else:
        combination = None
    if len(sys.argv) == 4:
        specific_celltype = sys.argv[3]
    else:
        specific_celltype = "all"
    dname = data_name.split('-')[0]

else:
    raise Exception("InValid data name")

adata = sc.read(f"./data/{dname}_{'normalized' if normalize else 'count'}_{combination}.h5ad")
adata = adata[adata.obs[condition_key].isin(conditions)]

if specific_celltype != 'all' and specific_celltype not in list(adata.obs[cell_type_key]):
    raise Exception(f"Specified celltype '{specific_celltype}' not found. Options: {list(adata.obs[cell_type_key].unique())}")

# if adata.shape[1] > 2000:
#     sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#     adata = adata[:, adata.var['highly_variable']]

#train_adata, valid_adata = reptrvae.utils.train_test_split(adata, 0.80)
train_adata = adata[adata.obs["Validation"] == "Train"]
valid_adata = adata[adata.obs["Validation"] == "Test"]

if normalize:
    params = {
        "lr": 0.00005,
        "epochs": 1000,
        "batch": 4,
        "earlyStop": 100,
        "lrReduce": 50
    }
else:
    params = {
        "lr": 0.001,
        "epochs": 50000,
        "batch": 4,
        "earlyStop": 300,
        "lrReduce": 150
    }

if specific_celltype == 'all':
    network = reptrvae.models.trVAE(x_dimension=train_adata.shape[1],
                                    z_dimension=40,
                                    n_conditions=len(train_adata.obs[condition_key].unique()),
                                    alpha=5e-5,
                                    beta=500,
                                    eta=100,
                                    clip_value=1e6,
                                    lambda_l1=0.0,
                                    lambda_l2=0.0,
                                    learning_rate=params['lr'],
                                    model_path=f"./models/trVAE/best/{data_name}-{specific_celltype}/",
                                    dropout_rate=0.2,
                                    output_activation='relu')

    network.train(train_adata,
                    valid_adata,
                    labelencoder,
                    condition_key,
                    n_epochs=params['epochs'],
                    batch_size=params['batch'],
                    verbose=2,
                    early_stop_limit=params['earlyStop'],
                    lr_reducer=params['lrReduce'],
                    shuffle=True,
                    save=False,
                    retrain=True,
                    )

    train_labels, _ = reptrvae.tl.label_encoder(train_adata, labelencoder, condition_key)
    mmd_adata = network.to_mmd_layer(train_adata, train_labels, feed_fake=-1)

    source_adata = adata[adata.obs[condition_key] == source_condition]
    target_adata = adata[adata.obs[condition_key] == target_condition]
    source_labels = np.zeros(source_adata.shape[0]) + labelencoder[source_condition]
    target_labels = np.zeros(source_adata.shape[0]) + labelencoder[target_condition]

    pred_adata = network.predict(source_adata,
                                    encoder_labels=source_labels,
                                    decoder_labels=target_labels,
                                    )

    pred_adata.obs[condition_key] = [f"{source_condition}_to_{target_condition}"] * pred_adata.shape[0]
    pred_adata.obs[cell_type_key] = specific_celltype

    adata_to_write = pred_adata.concatenate(target_adata)
    adata_to_write.write_h5ad(f"./data/reconstructed/trVAE_{data_name}/{specific_celltype}_{source_condition}_to_{target_condition}_{'norm' if normalize else 'count'}_{combination}.h5ad")
    # reptrvae.pl.plot_umap(mmd_adata,
    #                       condition_key, cell_type_key,
    #                       frameon=False, path_to_save=f"./results/{data_name}/", model_name="trVAE_MMD",
    #                       ext="png")
else:
    net_train_adata = train_adata[
        ~((train_adata.obs[cell_type_key] == specific_celltype) & (train_adata.obs[condition_key].isin(target_conditions)))]
    net_valid_adata = valid_adata[
        ~((valid_adata.obs[cell_type_key] == specific_celltype) & (valid_adata.obs[condition_key].isin(target_conditions)))]

    network = reptrvae.models.trVAE(x_dimension=net_train_adata.shape[1],
                                    z_dimension=40,
                                    n_conditions=len(net_train_adata.obs[condition_key].unique()),
                                    alpha=5e-5,
                                    beta=500,
                                    eta=100,
                                    clip_value=1e6,
                                    lambda_l1=0.0,
                                    lambda_l2=0.0,
                                    learning_rate=params['lr'],
                                    model_path=f"./models/trVAE/best/{data_name}-{specific_celltype}/",
                                    dropout_rate=0.2,
                                    output_activation='relu')

    network.train(net_train_adata,
                  net_valid_adata,
                  labelencoder,
                  condition_key,
                  n_epochs=params['epochs'],
                  batch_size=params['batch'],
                  verbose=2,
                  early_stop_limit=params['earlyStop'],
                  lr_reducer=params['lrReduce'],
                  shuffle=True,
                  save=False,
                  retrain=True,
                  )

    train_labels, _ = reptrvae.tl.label_encoder(net_train_adata, labelencoder, condition_key)
    mmd_adata = network.to_mmd_layer(net_train_adata, train_labels, feed_fake=-1)

    cell_type_adata = adata[adata.obs[cell_type_key] == specific_celltype]
    source_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]
    target_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
    source_labels = np.zeros(source_adata.shape[0]) + labelencoder[source_condition]
    target_labels = np.zeros(source_adata.shape[0]) + labelencoder[target_condition]

    pred_adata = network.predict(source_adata,
                                 encoder_labels=source_labels,
                                 decoder_labels=target_labels,
                                 )

    pred_adata.obs[condition_key] = [f"{source_condition}_to_{target_condition}"] * pred_adata.shape[0]
    pred_adata.obs[cell_type_key] = specific_celltype

    adata_to_write = pred_adata.concatenate(target_adata)
    adata_to_write.write_h5ad(f"./data/reconstructed/trVAE_{data_name}/{specific_celltype}_{source_condition}_to_{target_condition}_{'norm' if normalize else 'count'}_{combination}.h5ad")
    # reptrvae.pl.plot_umap(mmd_adata,
    #                       condition_key, cell_type_key,
    #                       frameon=False, path_to_save=f"./results/{data_name}/", model_name="trVAE_MMD",
    #                       ext="png")
