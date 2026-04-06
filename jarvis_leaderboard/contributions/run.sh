#!/bin/bash
pip install -q jarvis-tools scikit-learn tqdm pandas

jarvis_populate_data.py --benchmark_file AI-SinglePropertyPrediction-formula_energy-ssub-test-mae --output_path=Out --json_key formula --id_tag id

python -c "
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from jarvis.db.jsonutils import loadjson
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors

dataset_info = loadjson('Out/dataset_info.json')
df = pd.read_csv('Out/id_prop.csv', header=None, names=['formula', 'form_energy'])
df['id'] = df.index + 1

tqdm.pandas()
df['cfid_desc'] = df['formula'].progress_apply(lambda f: get_chem_only_descriptors(f)[0])

train_df = df[:dataset_info['n_train']]
test_df = df[dataset_info['n_train']:]

X_train = np.array(train_df['cfid_desc'].tolist())
y_train = train_df['form_energy'].values
X_test = np.array(test_df['cfid_desc'].tolist())

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

results = pd.DataFrame({
    'id': test_df['id'].values,
    'prediction': y_pred
})

filename = dataset_info['benchmark_file'] + '.csv'
results.to_csv(filename, index=False)

with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(filename, filename)

print('Saved', filename + '.zip')
"
