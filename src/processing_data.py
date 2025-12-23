import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


with open('config.yaml', 'r') as f:
    full_config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = full_config.get('data', {})

def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Split train/val first
    target_col = data_config['target']
    test_size = data_config.get('test_size', 0.2)
    random_state = data_config.get('random_state', 42)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # --- NUMERIC FEATURES ---
    num_config = data_config['number_features']
    num_features = num_config['features']
    missing_strategy = num_config.get('missing_values', 'mean')
    log_transform_flag = num_config.get('log_transform', False)
    
    X_train_num = X_train[num_features].copy()
    X_val_num = X_val[num_features].copy()
    
    # Handle missing values (only bmi)
    if missing_strategy != 'remove':
        imputer = SimpleImputer(strategy=missing_strategy)
        X_train_num[['bmi']] = imputer.fit_transform(X_train_num[['bmi']])
        X_val_num[['bmi']] = imputer.transform(X_val_num[['bmi']])
    else:
        # remove rows with missing bmi
        mask_train = X_train_num['bmi'].notna()
        mask_val = X_val_num['bmi'].notna()
        X_train_num = X_train_num[mask_train]
        y_train = y_train[mask_train]
        X_val_num = X_val_num[mask_val]
        y_val = y_val[mask_val]
        X_train = X_train[mask_train]
        X_val = X_val[mask_val]
    
    # Log transform if enabled (only avg_glucose_level and bmi)
    if log_transform_flag:
        for col in ['avg_glucose_level', 'bmi']:
            transformer = FunctionTransformer(func=lambda x: np.log1p(x))
            X_train_num[[col]] = transformer.fit_transform(X_train_num[[col]])
            X_val_num[[col]] = transformer.transform(X_val_num[[col]])
    
    # --- CATEGORICAL FEATURES ---
    cat_config = data_config['categorical_features']
    cat_features = cat_config['features']
    encoding = cat_config.get('encoding', 'one_hot')
    
    X_train_cat = X_train[cat_features].copy()
    X_val_cat = X_val[cat_features].copy()
    
    if encoding == 'one_hot':
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_enc = pd.DataFrame(
            ohe.fit_transform(X_train_cat),
            index=X_train_cat.index,
            columns=ohe.get_feature_names_out(cat_features)
        )
        X_val_cat_enc = pd.DataFrame(
            ohe.transform(X_val_cat),
            index=X_val_cat.index,
            columns=ohe.get_feature_names_out(cat_features)
        )
    elif encoding == 'label':
        X_train_cat_enc = X_train_cat.copy()
        X_val_cat_enc = X_val_cat.copy()
        for col in cat_features:
            le = LabelEncoder()
            X_train_cat_enc[col] = le.fit_transform(X_train_cat[col])
            X_val_cat_enc[col] = le.transform(X_val_cat[col])
    elif encoding == 'target':
        # target encoding
        X_train_cat_enc = X_train_cat.copy()
        X_val_cat_enc = X_val_cat.copy()
        for col in cat_features:
            # compute mean target per category
            mapping = y_train.groupby(X_train_cat[col]).mean()
            X_train_cat_enc[col] = X_train_cat[col].map(mapping)
            X_val_cat_enc[col] = X_val_cat[col].map(mapping).fillna(y_train.mean())
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    # Combine numeric + categorical
    X_train_processed = pd.concat([X_train_num, X_train_cat_enc], axis=1)
    X_val_processed = pd.concat([X_val_num, X_val_cat_enc], axis=1)
    
    return X_train_processed, X_val_processed, y_train, y_val


