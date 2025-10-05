import pickle
import os
from sklearn.decomposition import PCA

DATASET_NAME = "beauty"
PARENT = os.path.dirname
SCRIPT_DIR = PARENT(os.path.abspath(__file__))

llm_item_emb = pickle.load(open(os.path.join(SCRIPT_DIR, 'handle', 'item_emb_np.pkl'), "rb"))

pca = PCA(n_components=64)
pca_item_emb = pca.fit_transform(llm_item_emb)

with open(os.path.join(SCRIPT_DIR, 'handle', 'pca64_item_emb_np.pkl'), "wb") as f:
    pickle.dump(pca_item_emb, f)

# pca = PCA(n_components=128)
# pca_item_emb = pca.fit_transform(llm_item_emb)

# with open(os.path.join(SCRIPT_DIR, 'handle', 'pca128_itm_emb_np.pkl'), "wb") as f:
#     pickle.dump(pca_item_emb, f)