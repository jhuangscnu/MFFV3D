import numpy
import pandas as pd
import os
import sys
from sklearn import svm
import argparse


# function for getting the authentication socres from the results we have saved
def get_scores(path=None, csv=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        csv: name of the input csv file
    Returns:
        data['score']: authentication scores
    '''
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    return data['score']

    
# function for getting the authentication socres of all the three views
def get_all_scores(path=None, pt=None, model=None, st=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        pt: protocol, balance or normal
        st: set type, dev or test
        model: model name 
    Returns:
        all_scores: authentication scores of all the three views
    '''
    path_csv = path
    all_scores = []
    for ds in ['tex', 'sha']:
        csv = model+'_'+ds+'_'+pt+'_'+st + '_scores.csv'
        scores = get_scores(path=path_csv, csv=csv)
        all_scores.append(scores)
    return all_scores


# function for getting the finger ID
def get_id(path=None, name=None, pt=None, model=None, st=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        name: finger name, probe_subject_id or bio_ref_subject_id
        pt: protocol, balance or normal
        st: set type, dev or test
    Returns:
        data[name]: finger index
    '''
    csv = model+'_'+'tex'+'_'+pt+'_'+st + '_scores.csv'
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    return data[name]
    
# function for save the data to a csv file
def save_csv(csv_data, csv_path, csv_name, csv_title):
    '''
    Args:
        csv_data: data we want to save in csv file
        csv_path: path we want to save the csv file
        csv_name: name of the csv file we want to save
        csv_title: title in the csv file
    '''
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    # path_csv = csv_path + csv_name
    path_csv = os.path.join(csv_path, csv_name)
    csv = pd.DataFrame(columns=csv_title, data=csv_data)
    csv.to_csv(path_csv, index=False)


# function for getting the label for training the SVM
def get_svm_target(path=None, csv=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        csv: name of the csv file
    Returns:
        target: target for training SVM
    '''
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    target = []
    for idx in range(len(data)):
        if data['probe_subject_id'][idx] != data['bio_ref_subject_id'][idx]:
            target.append(0)
        else:
            target.append(1)
    return target



# function for SVM scores-fusion
def svm_fusion(path=None, kernel=None, model=None, pt=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        kernel: kernel of SVM
        pt: protocol
    Returns:
        scores_dev[:, 1]: SVM fusion scores of dev set
        scores_test[:, 1]: SVM fusion scores of test set
    '''
    all_scores_dev = get_all_scores(path=path, pt=pt, st='dev', model=model)
    all_scores_dev = numpy.array(all_scores_dev).transpose()

    all_scores_test = get_all_scores(path=path, pt=pt, st='test', model=model)
    all_scores_test = numpy.array(all_scores_test).transpose()

    csv_dev = model+'_'+'tex'+'_'+pt+ '_dev_scores.csv'
    target_dev = get_svm_target(path=path, csv=csv_dev)

    fusion = svm.SVC(kernel=kernel, probability=True)
    fusion.fit(all_scores_dev, target_dev)

    scores_dev = fusion.predict_proba(all_scores_dev)

    scores_test = fusion.predict_proba(all_scores_test) # use the SVM trained on dev set for test set
    return scores_dev[:, 1], scores_test[:, 1]
    


# function for saving the SVM fusion scores
def svm_score_fusion(path=None, path_save=None, kernel=None, csv_title=None, pt=None, model=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        path_save: path we want to save the fusion results
        kernel: kernel of SVM
        csv_title: title in the csv file
        pt: protocol
        model: model name
    '''
    probe_subject_id_dev = get_id(path=path, name='probe_subject_id', pt=pt, st='dev', model=model)
    bio_ref_subject_id_dev = get_id(path=path, name='bio_ref_subject_id', pt=pt, st='dev', model=model)

    probe_subject_id_test = get_id(path=path, name='probe_subject_id', pt=pt, st='test', model=model)
    bio_ref_subject_id_test = get_id(path=path, name='bio_ref_subject_id', pt=pt, st='test', model=model)

    scores_dev, scores_test = svm_fusion(path=path, kernel=kernel, pt=pt, model=model)

    data_save_dev = zip(*[numpy.array(probe_subject_id_dev), numpy.array(bio_ref_subject_id_dev), scores_dev])
    csv_save_dev = model+'_fusion_' + pt + '_dev_scores.csv'
    save_csv(csv_data=data_save_dev, csv_path=path_save, csv_name=csv_save_dev, csv_title=csv_title)

    data_save_test = zip(*[numpy.array(probe_subject_id_test), numpy.array(bio_ref_subject_id_test), scores_test])
    csv_save_test = model+'_fusion_' + pt + '_test_scores.csv'
    save_csv(csv_data=data_save_test, csv_path=path_save, csv_name=csv_save_test, csv_title=csv_title)

    
# function for scores fusion
def run_score_level_fusion(path_single=None, path_fusion=None, list_model=None, pt=None):

    for model in list_model: # ablation experiments
        for kernel in ['poly']:
            svm_score_fusion(path=path_single, path_save=path_fusion, kernel=kernel, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], model=model, pt=pt)
            print('model ', model, 'svm with kernel ', kernel, ' with ', pt,  ' done!')
    


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-rs','--path_rs', type=str, default='SNM/', help='path of results', required=False) # path of the results (csv files) of experiments fvia.py
    parser.add_argument('-fs','--path_fs', type=str, default='SNM/', help='path of fusion', required=False) # path we want to save the fusion scores
    parser.add_argument('-m', '--model', type=list, default=['fsnet_x'], help='model list', required=False)
    args = parser.parse_args()
    
    for pt in ['bal', 'nom']:
        run_score_level_fusion(path_single=args.path_rs, path_fusion=args.path_fs, list_model=args.model, pt=pt)

if __name__ == '__main__':
    main(sys.argv)






