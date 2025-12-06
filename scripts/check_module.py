# scripts/check_models.py
import joblib, os, sklearn
vec_path = 'models/vectorizer.pkl'
mod_path = 'models/model.pkl'
print('Files present:')
print(' - vectorizer:', os.path.exists(vec_path))
print(' - model:', os.path.exists(mod_path))
if os.path.exists(vec_path):
    v = joblib.load(vec_path)
    print('vectorizer type:', type(v))
    print('has vocabulary_:', hasattr(v,'vocabulary_'))
    print('has idf_:', hasattr(v,'idf_'))
    if hasattr(v,'vocabulary_'):
        print('vocabulary size:', len(v.vocabulary_))
if os.path.exists(mod_path):
    m = joblib.load(mod_path)
    print('model type:', type(m))
    print('has coef_:', hasattr(m,'coef_'))
    print('has classes_:', hasattr(m,'classes_'))
print('local sklearn version:', sklearn.__version__)
print('If vectorizer lacks vocabulary_/idf_, re-save the fitted vectorizer in Colab before downloading.')