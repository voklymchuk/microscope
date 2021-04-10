
DEBUG = False

paths, labels = getTrainDataset()

# divide to
keys = np.arange(paths.shape[0], dtype=np.int)
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])

if DEBUG == True:  # use only small subset for debugging, Kaggle's RAM is limited
    pathsTrain = paths[0:256]
    labelsTrain = labels[0:256]
    pathsVal = paths[lastTrainIndex : lastTrainIndex + 256]
    labelsVal = labels[lastTrainIndex : lastTrainIndex + 256]
    use_cache = True
else:
    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]
    use_cache = True

# print(paths.shape, labels.shape)
# print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

file_name = os.getcwd() + "/base.model"
# test_dataset = h5py.File(file_name, "r")


tg = ProteinDataGenerator(
    pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=use_cache
)
vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=use_cache)


# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint(
    file_name,
    monitor="val_f1",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    period=1,
)

epochs = 3

if DEBUG == True:
    use_multiprocessing = True  # DO NOT COMBINE WITH CACHE!
    workers = 1  # DO NOT COMBINE WITH CACHE!
else:
    use_multiprocessing = True
    workers = 1
