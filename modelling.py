from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle

def model(rfm_df):
    rfm_df = rfm_df.sample(frac=1)
    X = rfm_df[['recency', 'frequency', 'value', 'age']]
    y = rfm_df[['label']].values.reshape(-1)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    
    oversample = SMOTE()
    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    rf_over = RandomForestClassifier(n_estimators=10).fit(X_train_over, y_train_over)
    
    predictions = pd.DataFrame()
    predictions['true'] = y_train
    predictions['preds'] = rf.predict(X_train)
    
    predictions_test = pd.DataFrame()
    predictions_test['true'] = y_test
    predictions_test['preds'] = rf.predict(X_test)
    predictions_test['preds_over'] = rf_over.predict(X_test)
    
    train_acc = accuracy_score(predictions.true, predictions.preds)
    test_acc = accuracy_score(predictions_test.true, predictions_test.preds)
    test_acc_over = accuracy_score(predictions_test.true, predictions_test.preds_over)
    
    if(test_acc > test_acc_over):
        rf_final = rf
    else:
        rf_final = rf_over
        
    filepath = 'BCG_model.sav'
    pickle.dump(rf_final, open(filepath, 'wb'))
    
    feature_names = [i for i in X.columns]
    importances = rf_final.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_final.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    probs = rf_final.predict_proba(X_test)[:,1]
    
    return X_test, probs
    
    
