DIR = r"../data/raw/human-protein-atlas-image-classification"
# DIR = f"gs://{BUCKET}/data/human-protein-atlas-image-classification"
# BUCKET=os.getenv("BUCKET")
batch_size = 16
img_height = 200
img_width = 200
BATCH_SIZE = 16

SEED = 777
SHAPE = (512, 512, 4)

VAL_RATIO = 0.1  # 10 % as validation
DEBUG = True


THRESHOLD = np.float64(
    0.05
)  # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'
