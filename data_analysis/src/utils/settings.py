from dataclasses import dataclass 

@dataclass
class Settings():
    DATA_RUTE: str = r"data/"
    ORIGINAL_DATA_RUTE: str = r"data/original_data/"
    DATA_VERSION: int = 3
    FIRST_MOVIE_YEAR: int = 1888
    ACTUAL_YEAR: int = 2024
    
@dataclass
class Model_Settings():
    SEED: int = 42
    
    TEST_SET: int = 0
    TEST_PROPORTION: float = 0.2
    
    THRESHOLD_GENERAL: float = 0.5
    THRESHOLD_RIDGE: float = 0.03
    
    LAYER_SIZE: int = 64
    DENSE_SHAPE: bool = True
    EPOCHS: int = 50
    VERBOSE_INTERVAl: int = 1000
    PATIENCE: int = 5
    PATIENCE_EPS: float = 0.01
    
    KNN: int = 5
    