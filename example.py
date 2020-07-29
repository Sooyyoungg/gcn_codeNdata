from CustomCVs import StratifiedKFoldMixedSizes, StratifiedKFoldByGroups
from data_handling import create_data_set
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

# Here we will import our data from the csv: we do NOT drop out participants with incomplete features (complete=False), but they should have at least 90% feature completeness (completeness_threshold=0.9), we add Age and Sex and SiteID (age_group_tesla_site) as extra columns (covariates), and we automaticly exclude sites with too little participants (min_counts_per_site). I added documentation to this function, check it for further details.

X, y, groups = create_data_set(complete=False, completeness_threshold=0.9,
                               covariates=['Age', 'Sex', 'age_group_tesla_site'], min_counts_per_site='auto')


# Now we import the cross-validators we want to use, depending on the specific analysis we want to perform. These are based on scikit learn's CV classes (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

# Since I am performing gridsearches to optimize my models, I have to use a nested cross-validation loop (inner_cv).
# If you are not doing any nested cross-validation you can ignore the inner_cv's for now and only have to use the outer_cv lines.

random_seed = 0 # By setting this on zero we ensure we have the exact same splits!

# '1. Outer CV: Site-stratified fixed fold sizes, Inner CV: Site-stratified fixed fold sizes'
outer_cv = StratifiedKFoldByGroups(n_splits=10, random_state=random_seed, shuffle=True)
inner_cv = StratifiedKFoldByGroups(n_splits=5, random_state=random_seed, shuffle=True)

# '2. Outer CV: Leave One Group Out, Inner CV: Group K-Fold' 
outer_cv = LeaveOneGroupOut() # Note that here we don't have to use a random seed, as these splits will always be the same
inner_cv = GroupKFold(n_splits=5)

# '3. Outer CV: Site-stratified mixed fold sizes, Inner CV: Site-stratified fixed fold sizes'
outer_cv = StratifiedKFoldMixedSizes(random_state=random_seed)
inner_cv = StratifiedKFoldByGroups(n_splits=5, random_state=random_seed, shuffle=True)


# Then you can generate your CV splits like this:

for id_iter_cv, (train_id, test_id) in enumerate(outer_cv.split(X, y, groups)):

        print("Iteration: {}".format(id_iter_cv + 1))
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]
        groups_train, groups_test = groups[train_id], groups[test_id]

	# Do some fancy machine learning here.


# If you don't want to use any loops like this you could also extract the splits using something like this:

cv_splits = np.array(list(test_cv.split(X, y, groups)))

