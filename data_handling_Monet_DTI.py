import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# DTI data
#CSV_PATH=os.path.join('./dat/', "dti.merge13.csv")
#CSV_PATH=os.path.join('./dat/', "dti.complete.csv")
#CSV_PATH=os.path.join('./dat/', "dti.complete.zscore.csv")
#CSV_PATH=os.path.join('./dat/', "dti.zscore.harmonized.csv")
#CSV_PATH=os.path.join('./dat/', "dti.zscore.harmonized.cov_age.csv")

# harmonized
#CSV_PATH=os.path.join('./dat/Updated_data', "ENIGMA_DTI_harmo_1253obs_train.csv")
# non harmonized
#CSV_PATH=os.path.join('./dat/Updated_data', "ENIGMA_DTI_nonharmo_1253obs_train.csv")

# dai와 train set 동일시
#==========Dx==========
#--------Total---------
# non harmonized
#CSV_PATH=os.path.join('./dat/DTI_FINAL_DATA/Dx/total', "z_ENIGMA_DTI_nonharmo_1253obs_train.csv")
# harmonized
CSV_PATH=os.path.join('./dat', "z_ENIGMA_DTI_harmo_1253obs_train.csv")

#--------Adult---------
# non harmonized
#CSV_PATH=os.path.join('./dat/DTI_FINAL_DATA/Dx/Adult', "adult_z_ENIGMA_DTI_nonharmo_1223obs_train.csv")
# harmonized
#CSV_PATH=os.path.join('./dat/DTI_FINAL_DATA/Dx/Adult', "adult_z_ENIGMA_DTI_harmo_1223obs_train.csv")



def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def find_min_counts_per_site(counts):
    threshold = 0
    complete = False

    while not complete:
        n_selected = sum(counts > threshold)
        min_groups = min(counts[counts > threshold])

        # n_selected: threshold값보다 많이 사용된 site들 종류 개수
        # min_groups: threshold값보다는 많이 사용된 site 종류 중 가장 적게 사용된 site의 횟수
        if not min_groups < n_selected: # min_groups >= n_selected
            complete = True
        else:
            threshold += 1

    return threshold


def create_data_set(TesTrain = "Train", complete=False, completeness_threshold=0.9, age_group=None, covariates=None, query_str=None, min_counts_per_site=None, csv_path=CSV_PATH, y_label=None, verbose=False):
    """
    Create ENIGMA data set for classification with given constraints.

    :param complete : boolean, default=False
        Return data without missing values by dropping all rows that contain any NA entry.
    :param completeness_threshold : float, default=0.9
        Specifies threshold for the minimum proportion of complete entries that each row should have (will only be in
        effect if complete = False). Threshold should be in the range between 0. (no threshold is used) and 1. (drop
        rows that contain any NA entry).
    :param age_group : str, default=None
        Limit data set to specific age group (either "1_adult", " 2_pediatric" or None to include all participants).
    :param covariates : str or list, default=None
        Covariates that will be returned from the data set.
    :param query_str: str, default=None
        Query used to filter ENIGMA data frame for specific sample (e.g. 'Sex ==  1 & Med == 1')
    :param min_counts_per_site: int or str, default=None
        Specifies threshold of the minimum sample size per site (either fixed by passing an int, found automatically by
        using "auto" or None by default).
    :param csv_path : str, default=CSV_PATH
        Path to ENIGMA csv file. Default is set in CSV_PATH global.
    :param y_label : str, default=None,
	    Label for custom variable that will be returned as y. Set to "Dx" (OCD diagnosis) by default.
    :param verbose : boolean, default False 
        -> verbose 값을 False로 지정하기 때문에 모든 관련 코드 삭제
        Enable prints for detailed logging information.
    :return: X, selected_fs_features, C, y, groups
    """

    if(TesTrain == "Test"):
        print("test set 설정")
        #csv_path=os.path.join('./dat/Updated_data', "ENIGMA_DTI_harmo_1253obs_test.csv")
        #csv_path=os.path.join('./dat/Updated_data', "ENIGMA_DTI_nonharmo_1253obs_test.csv")
        
        #==========Dx==========
        #--------Total---------
        # non harmonized
        #csv_path=os.path.join('./dat/DTI_FINAL_DATA/Dx/total', "z_ENIGMA_DTI_nonharmo_1253obs_test.csv")
        # harmonized
        csv_path=os.path.join('./dat', "z_ENIGMA_DTI_harmo_1253obs_test.csv")

        #--------Adult---------
        # non harmonized
        #csv_path=os.path.join('./dat/DTI_FINAL_DATA/Dx/Adult', "adult_z_ENIGMA_DTI_nonharmo_1223obs_test.csv")
        # harmonized
        #csv_path=os.path.join('./dat/DTI_FINAL_DATA/Dx/Adult', "adult_z_ENIGMA_DTI_harmo_1223obs_test.csv")

    df = load_csv(csv_path)

    print('Loading ENIGMA dataset: Complete = {}, Min_Threshold = {}, Covariates = {}, ''Min_Counts_per_Site = {}, y_label \n'.format(complete, completeness_threshold, covariates, min_counts_per_site, y_label))

    # 구조물 사이의 tract 열이 시작하는 위치
    #first_fs_index = np.where(df.columns.values == 'ACR.fa')[0][0] -> 예전 data
    first_fs_index = np.where(df.columns.values == 'ACR.FA')[0][0]

    # index = 4, 6 -> Age, Sex column
    #fs_labels = []
    #fs_labels.append(df.columns.values[4])
    #fs_labels.append(df.columns.values[6])
    fs_labels = df.columns.values[range(first_fs_index, len(df.columns) - 1 - 4)]
    cov_labels = np.append(df.columns.values[range(0, first_fs_index)], ["site", "Dx"])

    y_label = y_label if y_label else "Dx"

    
    # 1) covariates = [site]
    # 2) covariates = [Age, Sex, site]
    if covariates:
        if isinstance(covariates, str):
            covariates = [covariates]
        if not isinstance(covariates, list):
            raise RuntimeError('TypeError: covariates must be str or list, {} not accepted \n'.format(type(covariates)))
        if not set(covariates).issubset(set(cov_labels)):
            raise RuntimeError('Warning! Unknown covariates specified: {} \n'
                               'Only the following options are allowed: {} \n'.format(covariates, cov_labels))
        else:
            all_selected_features = np.append(fs_labels, covariates)
    else:
        all_selected_features = fs_labels


    if complete:
        # Remove subjects that miss ANY feature: Dx + fs_labels + covariates 열에서 결측치 제거
        tmp_df = df[~df.loc[:, np.append(y_label, all_selected_features)].T.isnull().any()]
        #print("tmp_df:", len(tmp_df))   # 결측치 제거한 후 행: 651개 (700개 행에서 결측치 존재)
    else:
        # Remove subjects that miss target label
        n_ = df.shape[0]
        tmp_df = df[~df.loc[:, [y_label]].T.isnull().any()]

        # Remove subjects which miss more features than given threshold
        nans_per_subject = tmp_df.loc[:, all_selected_features].isnull().T.sum()
        completeness_per_subject = (float(tmp_df.shape[1]) - nans_per_subject) / tmp_df.shape[1]
        tmp_df = tmp_df.loc[completeness_per_subject >= completeness_threshold]
        n_dropped = sum(completeness_per_subject < completeness_threshold)

    # query_str = False (default)
    if query_str:
        try:
            n_org = tmp_df.shape[0]
            tmp_df = tmp_df.query(query_str)
            n_fil = tmp_df.shape[0]
        except RuntimeError:
            raise

    # age_group_tesla_site -> site
    groups = tmp_df.loc[:, "site"].values  # 모든 행들의 site 열 값들만 저장 (site 이름)
    sites, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    tmp_df.loc[:, "site"] = inverse  # site 종류에 따라 0~6 값으로 치환
    #print("tmp_df: ", len(tmp_df)) # 651
    
    #print(len(tmp_df)) # test_dtiharmonized_cov.age.csv에서의 결과: 250
    
    X = tmp_df.loc[:, fs_labels]
    print("X: ", len(X)) #651
    y = tmp_df.loc[:, y_label].values.astype(int)
    #print("y: ", y) # 0과 1의 값으로 저장
    C = tmp_df.loc[:, covariates].values if covariates else np.array([]).reshape((X.shape[0], 0)) # site만 cov로 설정하면 한 행당 하나의 값만 저장
    #print("covariates:",covariates)
    #print(C)

    # (Hard)code y labels for now
    if y_label == "Med":
        y = y - 1
    elif y_label == "Sev":
        y = np.array(y > 24, dtype=int)
    elif y_label == "Dur":
        y = np.array(y > 7, dtype=int)
    elif y_label == "AO":
        y = np.array(y >= 18, dtype=int)

    threshold = 0
    #print("counts:", counts)

    if min_counts_per_site:
        if min_counts_per_site == 'auto':
            threshold = find_min_counts_per_site(counts)
            #print("th:", threshold)
        elif isinstance(min_counts_per_site, int):
            threshold = min_counts_per_site
        else:
            raise RuntimeError('Invalid option for min_counts_per_site: {}, only integers and "auto" are allowed'.format(min_counts_per_site))
    
    included_sites = sites[counts > threshold]
    #print("sites:", included_sites)
    mask = np.isin(groups, included_sites)
    #print(mask)

    X, C, y, groups = X[mask], C[mask], y[mask], groups[mask]
    
    print('Finished loading data set: {} samples, {} FS features, {} covariates \n \n '.format(X.shape[0], X.shape[1], C.shape[1] if covariates else 0))

    return X, fs_labels, C, y, groups
