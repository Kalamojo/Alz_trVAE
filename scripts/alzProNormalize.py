import sys
import scanpy as sc

combination = sys.argv[1]
if combination not in ["I", "U"]:
    raise Exception("Invalid Combination Method")

adata = sc.read(f"./data/alzPro_count_{combination}.h5ad")
sc.pp.normalize_total(adata, target_sum=1000)
adata.write_h5ad(f"./data/alzPro_normalized_{combination}.h5ad")
