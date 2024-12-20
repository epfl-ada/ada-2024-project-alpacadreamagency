from dataclasses import dataclass 

@dataclass
class Settings():
    DATA_RUTE = r"data/"
    ORIGINAL_DATA_RUTE = r"data/original_data/"
    DATA_VERSION = 3
    FIRST_MOVIE_YEAR = 1888
    ACTUAL_YEAR = 2024
    
@dataclass
class Model_Settings():
    SEED = 42
    
    TEST_SET = 0
    TEST_PROPORTION = 0.2
    
    THRESHOLD_GENERAL = 0.5
    THRESHOLD_RIDGE = 0
    
    LAYER_SIZE = 64
    DENSE_SHAPE = True
    EPOCHS = 50
    VERBOSE_INTERVAl = 1000
    PATIENCE = 5
    PATIENCE_EPS = 0.025
    
    KNN = 5
    