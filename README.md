## David R. Winer
## Machine Learning Project (Fall 2017)

# Algorithms: 
- ID3.py : binary, multi-class, and bagging
- NAIVE_BAYES.py : multi-class
- ADABOOST.py : binary and multi-class
- Off-the-shelf LIBLINEAR ("/LogRegr_off_the_shelf")

# Organize Data:
- feature_engineering.py : compiles data from initial data source (WESTERN_DUEL_CROPUSMASTER_update..pkl), stores in "/Initial_dataObservations"
- SceneDataStructs.py : container script for observation data, also uses Entities.py
- split_into_test_and_folds.txt 
- generate_features.py : creates feature ids from "/Initial_dataObservations"
- background_check.py : counts shot foreground and background observations (zeta), unused in project

# Extracted knowledge:
- action types (action_types.txt)
- scene names (scenes.txt)
- shot durations (duration.txt)

# Other Notes:
- Implemented with Python 3.x (x>=3)