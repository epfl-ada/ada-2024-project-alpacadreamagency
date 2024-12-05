class Settings():
    DATA_RUTE = r"data/"
    ORIGINAL_DATA_RUTE = r"data/original_data/"
    DATA_VERSION = 3
    FIRST_MOVIE_YEAR = 1888
    ACTUAL_YEAR = 2024

class Model_Settings():
    SEED = 42
    
    TEST_SET = 0
    TEST_PROPORTION = 0.2
    
    THRESHOLD_GENERAL = 0.5
    THRESHOLD_RIDGE = 0
    
    LAYER_SIZE = 64
    EPOCH_SIZE = 5
    
    KNN = 5
    