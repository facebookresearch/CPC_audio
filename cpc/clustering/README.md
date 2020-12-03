# Clustering

## Estimate the clusters of the CPCs embeddings on a given dataset

Just run:
```
python clustering_script.py $PATH_CPC_CHECKPOINT \
                            $OUTPUT_DIRECTORY \
                            $DATASET_ROOT_DIR \
                            -k $N_CLUSTERS
```

You can explore the --DPMean option to try out the DPMean clustering.

## Perform the quantization of a dataset using precomputed clusters

```
python clustering_quantization.py $PATH_CLUSTERING_CHEKPOINT $DATASET_ROOT_DIR
```

Where PATH_CLUSTERING_CHEKPOINT refers to the .pt files containing the clusters.
