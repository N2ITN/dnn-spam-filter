#%%
# import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from tokenize_emails import model, X
embedding = X
model.load_weights("weights.best.hdf5")
#%%
model.get_config()
#%%
model.summary()
#%%
emb = model.layers[0]
dir(emb)
emb.get_weights()[0][0]
