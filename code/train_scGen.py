import anndata
import scanpy as sc
import scgen
from scipy import sparse


def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type_to_train=None):
    
    stim_keys = ["Hpoly.Day3", "Hpoly.Day10", "Salmonella"]
    ctrl_key = "Control"
    cell_type_key = "cell_label"
    train = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    valid = sc.read(f"../data/{data_name}/valid_{data_name}.h5ad")

    for cell_type in train.obs[cell_type_key].unique().tolist():
        if cell_type_to_train is not None and cell_type != cell_type_to_train:
            continue
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key].isin(stim_keys)))]
        net_valid_data = valid[~((valid.obs[cell_type_key] == cell_type) & (valid.obs[condition_key].isin(stim_keys)))]
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")

        network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size,
                     verbose=True, early_stop_limit=5)
        network.sess.close()
        print(f"network_{cell_type} has been trained!")


def reconstruct_whole_data(data_name="pbmc", condition_key="condition", cell_type_to_predict=None):
    stim_key = "Hpoly.Day10"
    ctrl_key = "Control"
    cell_type_key = "cell_label"
    train = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")

    all_data = anndata.AnnData()
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        if cell_type_to_predict is not None and cell_type != cell_type_to_predict:
            continue
        print(f"Reconstructing for {cell_type}")
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        pred, delta = network.predict(adata=net_train_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
        network.sess.close()
    all_data.write_h5ad(f"../data/reconstructed/scGen/{data_name}.h5ad")


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out("haber", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="Tuft")
    
    reconstruct_whole_data("haber")

