

def getTrainDataset():

    path_to_train = DIR + "/train/"
    data = pd.read_csv(DIR + "/train.csv")

    paths = []
    labels = []

    for name, lbl in zip(data["Id"], data["Target"].str.split(" ")):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def getTestDataset():

    path_to_test = DIR + "/test/"
    data = pd.read_csv(DIR + "/sample_submission.csv")

    paths = []
    labels = []

    for name in data["Id"]:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


class ProteinDataGenerator(keras.utils.Sequence):
    def __init__(
        self, paths, labels, batch_size, shape, shuffle=False, use_cache=False
    ):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def len(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        try:
            # print(path)
            R = Image.open(path + "_red.png")
            G = Image.open(path + "_green.png")
            B = Image.open(path + "_blue.png")
            Y = Image.open(path + "_yellow.png")
        except:
            o = urlparse(path)
            storage_client = storage.Client("home-225723")
            bucket = storage_client.get_bucket(o.netloc)
            # print(path)
            R = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_red.png").download_as_string())
            )
            G = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_green.png").download_as_string())
            )
            B = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_blue.png").download_as_string())
            )
            Y = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_yellow.png").download_as_string())
            )

        im = np.stack((np.array(R), np.array(G), np.array(B), np.array(Y)), -1)

        im = np.divide(im, 255)

        return im
