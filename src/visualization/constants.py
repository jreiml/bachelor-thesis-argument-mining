DEFAULT_FONT_SIZE = 14

COL_TOPIC = "topic"
COL_SENTENCE = "sentence"
COL_SET = "set"
COL_IS_ARGUMENT = "is_argument"
COL_IS_ARGUMENT_PROB = "is_argument_prob"
COL_IS_STRONG = "is_strong"
COL_IS_STRONG_PROB = "is_strong_prob"
COL_STANCE = "stance"
COL_STANCE_CONF = "stance_conf"

SET_TRAIN = "train"
SET_DEV = "dev"
SET_TEST = "test"


def get_in_topic_dataset_name(dataset_name):
    return dataset_name + " (In-Topic)"


def get_cross_topic_dataset_name(dataset_name):
    return dataset_name + " (Cross-Topic)"
