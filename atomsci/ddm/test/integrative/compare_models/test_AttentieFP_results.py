import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.pipeline.parameter_parser as pp
from atomsci.ddm.pipeline.compare_models import nan
import sys
import os
import shutil
import tarfile
import json
import glob
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../delaney_Panel'))
from test_delaney_panel import init, train_and_predict

sys.path.append(os.path.join(os.path.dirname(__file__), '../dc_models'))
from test_retrain_dc_models import H1_curate
from atomsci.ddm.utils import llnl_utils

def clean():
    delaney_files = glob.glob('delaney-processed*.csv')
    for df in delaney_files:
        if os.path.exists(df):
            os.remove(df)

    h1_files = glob.glob('H1_*.csv')
    for hf in h1_files:
        if os.path.exists(hf):
            os.remove(hf)

    if os.path.exists('result'):
        shutil.rmtree('result')

    if os.path.exists('scaled_descriptors'):
        shutil.rmtree('scaled_descriptors')

def get_tar_metadata(model_tarball):
    tarf_content = tarfile.open(model_tarball, "r")
    metadata_file = tarf_content.getmember("./model_metadata.json")
    ext_metadata = tarf_content.extractfile(metadata_file)

    meta_json = json.load(ext_metadata)
    ext_metadata.close()

    return meta_json

def confirm_perf_table(json_f, df):
    '''
    df should contain one entry for the model specified by json_f
    checks to see if the parameters extracted match what's in config
    '''
    # should only have trained one model
    assert len(df) == 1
    # the one row
    row = df.iloc[0]

    with open(json_f) as f:
        config = json.load(f)

    model_type = config['model_type']
    if model_type == 'NN':
        assert row['best_epoch'] >= 0
        assert row['max_epochs'] == int(config['max_epochs'])
        assert row['learning_rate'] == float(config['learning_rate'])
        assert row['layer_sizes'] == config['layer_sizes']
        assert row['dropouts'] == config['dropouts']
    elif model_type == 'RF':
        print(row[[c for c in df.columns if c.startswith('rf_')]])
        assert row['rf_estimators'] == int(config['rf_estimators'])
        assert row['rf_max_features'] == int(config['rf_max_features'])
        assert row['rf_max_depth'] == int(config['rf_max_depth'])
    elif model_type == 'xgboost':
        print(row[[c for c in df.columns if c.startswith('xgb_')]])
        assert row['xgb_gamma'] == float(config['xgb_gamma'])
        assert row['xgb_learning_rate'] == float(config['xgb_learning_rate'])
    else:
        assert model_type in pp.model_wl
        assert row['best_epoch'] >= 0
        pparams = pp.wrapper(config)
        assert row['learning_rate'] == float(pparams.learning_rate)

def compare_dictionaries(ref, model_info):
    '''
    Args:
        ref: this is the hardcoded reference dictionary. Everything in this
            dictionary must appear in output and they must be exactly the same or,
            if it's a numeric value, must be within 1e-6.
        model_info: This is the output from get_bset_perf_table
    Returns:
        None
    '''
    for k, v in ref.items():
        if not v  == v:
            # in the case of nan
            assert not model_info[k] == model_info[k]
        elif v is None:
            assert model_info[k] is None
        elif type(v) == str:
            assert model_info[k] == v
        else:
            # some kind of numerical object
            assert abs(model_info[k]-v) < 1e-6

def all_similar_tests(json_f, prefix='delaney-processed'):
    train_and_predict(json_f, prefix=prefix)

    df1 = cm.get_filesystem_perf_results('result', 'regression')
    confirm_perf_table(json_f, df1)

    df2 = cm.get_summary_perf_tables(result_dir='result', prediction_type='regression')
    confirm_perf_table(json_f, df2)

    model_uuid = df2['model_uuid'].values[0]
    model_info = cm.get_best_perf_table(metric_type='r2_score', model_uuid=model_uuid, result_dir='result')
    print('model_info:', model_info)
    confirm_perf_table(json_f, pd.DataFrame([model_info]))

    assert model_info['model_parameters_dict'] == df1.iloc[0]['model_parameters_dict']
    assert model_info['model_parameters_dict'] == df2.iloc[0]['model_parameters_dict']

    return df1, df2, model_info

def test_AttentiveFP_results():
    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_AttentiveFPModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {
        "max_epochs": 5,
        "AttentiveFPModel_mode":"regression",
        "AttentiveFPModel_num_layers":3,
        "AttentiveFPModel_learning_rate": 0.0007,
        "AttentiveFPModel_model_dir": "result",
        "AttentiveFPModel_n_tasks": 1,}

    assert json.loads(model_info['feat_parameters_dict']) == {"MolGraphConvFeaturizer_use_edges":"True",}

if __name__ == '__main__':
    test_AttentiveFP_results()