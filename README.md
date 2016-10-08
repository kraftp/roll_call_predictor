This is a code and data release for the paper "An Embedding Model for Predicting Roll-Call Votes" presented by Peter Kraft, Hirsh Jain, and Alexander Rush in EMNLP 2016.

Dependencies for preprocessing are Python with numpy, scikit-learn and the h5py library.  Dependencies for the model are Lua/Torch with the hdf5 and nn packages.

To generate a dataset, begin by downloading a particular Congress's bill information from [govtrack.us](govtrack.us).  Their bulk data download documentation is available [here](govtrack.us/developers/data).  If experimenting on Congresses after the 114th, you must also download the latest versions of legislators-current.csv and legislators-historic.csv from govtrack.  After downloading, preprocess a govtrack dataset in folder DATA as follows:

    python preprocess.py DATA

This will create DATA.hdf5 containing the preprocessed dataset.  It will also create a file called words.txt containing the words in the bills in order by index (for cross-reference with later word embeddings) as well as a file called cp_info.txt containing a list of congresspeople and their parties in order by index (for cross-referencing with later ideal vectors).

If you do not wish to generate your own datasets, we provide for replication purposes the six original hdf5 datasets we used to generate our published results in Data/original_splits.zip.

The model has the following controllable parameters:

-nepochs:  Number of epochs to run for.

-dp:  Size of interior dot product and congressperson embedding.

-eta:  Learning rate.


Given a datafile, to run the model using the settings used in the paper, run:

    th model.lua -datafile DATAFILE.hdf5 -classifier nn_embed_m -eta 0.1 -nepochs 10 -dp 10

Alternatively, if you want to generate ideal vectors or word embeddings for your own experimentation, run, changing the hyperparameters as necessary:

    th model.lua -datafile DATAFILE.hdf5 -classifier nn_embed_m_nocv -eta 0.1 -nepochs 10 -dp 10

This will tell the model to output a file named cp_weights.txt containing the ideal vectors of each congressperson as well as a file named bill_weights.txt containing the transformed word embeddings of each word in the dataset.
