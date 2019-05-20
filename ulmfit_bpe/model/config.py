import os

VERSION = os.environ.get("ULMFIT_VERSION", "version_3")

VERSION_FOLDER = "/tmp/{}/".format(VERSION)
COMMENTS_BPE_PATH = "comments.model"
DATA_PATH = "data"
MODEL_PATH = "comments_model"
DROPOUT_COEFFICIENT = 0.3
