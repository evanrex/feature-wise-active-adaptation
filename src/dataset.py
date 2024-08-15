from functools import partial
import itertools
from tkinter.messagebox import NO

from _config import *
from _shared_imports import *
import scipy.io as spio
import torchvision
from sys import exit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.io



def load_csv_data(path, labels_column=-1):
	"""
	Load a data file
	- path (str): path to csv_file
	- labels_column (int): indice of the column with labels
	"""
	Xy = pd.read_csv(path, index_col=0)
	X = Xy[Xy.columns[:labels_column]].to_numpy()
	y = Xy[Xy.columns[labels_column]].to_numpy()

	return X, y

def load_mice(args, one_hot = False):
    """
    Loading mice protein dataset 
    Higuera,Clara, Gardiner,Katheleen, and Cios,Krzysztof. (2015). Mice Protein Expression. UCI Machine Learning Repository. https://doi.org/10.24432/C50S3Z.

	Data processing (imputation and normalization) is done as in the CAE paper: Abid, Abubakar, Muhammad Fatih Balin, and James Zou. "Concrete autoencoders for differentiable feature selection and reconstruction." arXiv preprint arXiv:1901.09346 (2019).
    """
    filling_value = -100000
    
    mice_path = os.path.join(args.data_dir, 'Mice_Protein', 'Data_Cortex_Nuclear.csv')

    X = np.genfromtxt(mice_path, delimiter = ',', skip_header = 1, usecols = range(1, 78), filling_values = filling_value, encoding = 'UTF-8')
    classes = np.genfromtxt(mice_path, delimiter = ',', skip_header = 1, usecols = range(78, 81), dtype = None, encoding = 'UTF-8')

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean([X[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])])

    DY = np.zeros((classes.shape[0]), dtype = np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ['Control', 'Memantine', 'C/S'])):
            DY[i] += (2 ** j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]
    
    if not one_hot:
        Y = DY
        
    X = X.astype(np.float32)
    # Y = Y.astype(np.int32)
    
    return X, Y

def load_ASU_dataset(args, dataset):
	mat = scipy.io.loadmat(os.path.join(args.data_dir, "ASU_datasets", f"{dataset}.mat"))
	X = mat['X']
	y = np.squeeze(mat['Y'])
	X = X.astype(np.float64)
	y = y.astype(np.int64)

	if y.min() == 1 and y.max() == len(set(y)):
		y -= 1
	
	if y.min() == -1 and y.max() == 1 and len(set(y)) == 2:
		y = (y + 1) // 2

	return X, y	

def load_PBMC(args):
	X = np.loadtxt(os.path.join(args.data_dir, "PBMC_X.csv"), delimiter=",")
	y = np.loadtxt(os.path.join(args.data_dir, "PBMC_y.csv"), delimiter=",")
 
	X = X.astype(np.float32)
	y = y.astype(np.int64)
 
	assert X.shape == (2075, 21932)
	assert y.shape == (2075,)
 
	return X, y

def load_PBMC_small(args):
	X = np.loadtxt(os.path.join(args.data_dir, "PBMC_small_X.csv"), delimiter=",")
	y = np.loadtxt(os.path.join(args.data_dir, "PBMC_small_y.csv"), delimiter=",")
 
	X = X.astype(np.float32)
	y = y.astype(np.int64)
 
	return X, y
	
def load_finance(args):
	data = pd.read_csv(f'{args.data_dir}/Finance/finance.csv')
	X = data.drop(columns=['Class'])
	Y = data['Class']
	# convert dtype of Y to int
	Y = Y.astype(int)
	return X, Y

class CustomPytorchDataset(Dataset):
	def __init__(self, X, y, transform=None) -> None:
		# X, y are numpy
		super().__init__()

		self.X = torch.tensor(X, requires_grad=False)
		self.y = torch.tensor(y, requires_grad=False)
		self.transform = transform

	def __getitem__(self, index):
		x = self.X[index]
		y = self.y[index]
		if self.transform:
			x = self.transform(x)
			y = y.repeat(x.shape[0]) # replicate y to match the size of x

		return x, y

	def __len__(self):
		return len(self.X)


def standardize_data(X_train, X_valid, X_test, preprocessing_type):
	if preprocessing_type == 'standard':
		scaler = StandardScaler()
	elif preprocessing_type == 'minmax':
		scaler = MinMaxScaler()
	elif preprocessing_type == 'raw':
		scaler = None
	else:
		raise Exception("preprocessing_type not supported")

	if scaler:
		X_train = scaler.fit_transform(X_train).astype(np.float32)
		X_valid = scaler.transform(X_valid).astype(np.float32)
		X_test = scaler.transform(X_test).astype(np.float32)

	return X_train, X_valid, X_test


def compute_stratified_splits(X, y, cv_folds, seed_kfold, split_id):
	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed_kfold)
	
	for i, (train_ids, test_ids) in enumerate(skf.split(X, y)):
		if i == split_id:
			return X[train_ids], X[test_ids], y[train_ids], y[test_ids]


###############    EMBEDDINGS     ###############

def compute_histogram_embedding(args, X, embedding_size):
	"""
	Compute embedding_matrix (D x M) based on the histograms. The function implements two methods:

	DietNetwork
	- Normalized bincounts for each SNP

	FsNet
	0. Input matrix NxD
	1. Z-score standardize each column (mean 0, std 1)
	2. Compute the histogram for every feature (with density = False)
	3. Multiply the histogram values with the bin mean

	:param (N x D) X: dataset, each row representing one sample
	:return np.ndarray (D x M) embedding_matrix: matrix where each row represents the embedding of one feature
	"""
	X = np.rot90(X)
	
	number_features = X.shape[0]
	embedding_matrix = np.zeros(shape=(number_features, embedding_size))

	for feature_id in range(number_features):
		feature = X[feature_id]

		hist_values, bin_edges = np.histogram(feature, bins=embedding_size) # like in FsNet
		bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
		embedding_matrix[feature_id] = np.multiply(hist_values, bin_centers)

	return embedding_matrix

def compute_svd_embeddings(X, rank=None):
	"""
	- X (N x D)
	- rank (int): rank of the approximation (i.e., size of the embedding)
	"""
	assert type(X)==torch.Tensor
	assert X.shape[0] < X.shape[1]

	U, S, Vh = torch.linalg.svd(X, full_matrices=False)

	V = Vh.T

	if rank:
		S = S[:rank]
		V = V[:rank]

	return V, S

###############    DATASETS     ###############

class DatasetModule(pl.LightningDataModule):
	def __init__(self, args, X_train, y_train, X_valid, y_valid, X_test, y_test):
		super().__init__()
		self.args = args

		args.num_features = X_train.shape[1]
		args.num_classes = len(set(y_train).union(set(y_valid)).union(set(y_test)))

		# Standardize data
		self.X_train_raw = X_train
		self.X_valid_raw = X_valid
		self.X_test_raw = X_test

		X_train, X_valid, X_test = standardize_data(X_train, X_valid, X_test, args.patient_preprocessing)
		
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.X_test = X_test
		self.y_test = y_test
  
		if args.retrain_feature_selection:
			self.gen_selected_datasets(args.feature_importance)
			self.X_train = self.X_train_selected
			self.X_valid = self.X_valid_selected	
			self.X_test = self.X_test_selected
			args.num_features = self.X_train.shape[1]
   
		self.train_dataset = CustomPytorchDataset(X_train, y_train)
		self.valid_dataset = CustomPytorchDataset(X_valid, y_valid)
		self.test_dataset = CustomPytorchDataset(X_test, y_test)

		self.args.train_size = X_train.shape[0]
		self.args.valid_size = X_valid.shape[0]
		self.args.test_size = X_test.shape[0]

		# store the names of the validation dataloaders. They are appended to validation metrics
		self.val_dataloaders_name = [""] # the valid dataloader with the original data has no special name

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)

	def val_dataloader(self):
		# dataloader with original samples
		dataloaders = [DataLoader(self.valid_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)]

		# dataloaders for each validation augmentation type
		if self.args.valid_aug_dropout_p:
			if self.args.valid_aug_times==None:
				raise Exception("You must supply a list of --valid_aug_times.")

			# define some transformations
			def dropout_transform(x, p):
				"""
				- x (tensor): one datapoint
				- p (float): probability of droppint out features
				"""
				return F.dropout(x, p, training=True)

			def multiplicity_transform(x, no_times, transform):
				"""
				Args:
				- no_times (int): number of times to apply the transformation `transform`
				- transform (function): the tranformation function to be applied `no_times` times
				
				Return
				- stacked 2D tensor with the original samples and its augmented versions
				"""
				samples = [transform(x) for _ in range(no_times)]
				samples.append(x)

				return torch.stack(samples)

			for dropout_p, aug_times in itertools.product(self.args.valid_aug_dropout_p, self.args.valid_aug_times):
				partial_dropout_transform = partial(dropout_transform, p=dropout_p)
				print(f"Create validation dataset with dropout_p={dropout_p} and aug_times={aug_times}")
				partial_multiplicity_transform = partial(multiplicity_transform, no_times=aug_times, transform=partial_dropout_transform)
				
				valid_dataset_augmented = CustomPytorchDataset(self.X_valid, self.y_valid, transform=torchvision.transforms.Compose([
					torchvision.transforms.Lambda(partial_multiplicity_transform)
				]))

				dataloaders.append(DataLoader(valid_dataset_augmented, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory))
				self.val_dataloaders_name.append(f'dropout_p_{dropout_p}__aug_times_{aug_times}')
		
		print(f"Created {len(dataloaders)} validation dataloaders.")
		self.args.val_dataloaders_name = self.val_dataloaders_name # save the name to args, to be able to access them in the model (and use for logging)
		return dataloaders

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)

	def get_embedding_matrix(self, embedding_type, embedding_size):
		"""
		Return matrix D x M

		Use a the shared hyper-parameter self.args.embedding_preprocessing.
		"""
		if embedding_type == None:
			return None
		else:
			if embedding_size == None:
				raise Exception()

		# Preprocess the data for the embeddings
		if self.args.embedding_preprocessing == 'raw':
			X_for_embeddings = self.X_train_raw
		elif self.args.embedding_preprocessing == 'standard':
			X_for_embeddings = StandardScaler().fit_transform(self.X_train_raw)
		elif self.args.embedding_preprocessing == 'minmax':
			X_for_embeddings = MinMaxScaler().fit_transform(self.X_train_raw)
		else:
			raise Exception("embedding_preprocessing not supported")

		if embedding_type == 'histogram':
			"""
			Embedding similar to FsNet
			"""
			embedding_matrix = compute_histogram_embedding(self.args, X_for_embeddings, embedding_size)
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='all_patients':
			"""
			A gene's embedding are its patients gene expressions.
			"""
			embedding_matrix = np.rot90(X_for_embeddings)[:, :embedding_size]
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='svd':
			# Vh.T (4160 x rank) contains the gene embeddings on each row
			U, S, Vh = torch.linalg.svd(torch.tensor(X_for_embeddings, dtype=torch.float32), full_matrices=False) 
			
			Vh.T.requires_grad = False
			return Vh.T[:, :embedding_size].type(torch.float32)
		else:
			raise Exception("Invalid embedding type")

	def gen_MCAR_datasets(self, fraction = 0.1, seed = 0, replace_val = 0, include_train = False):
		"""
		Missing data mechanism
		- fraction (float): percentage of missing values
		- seed (int): seed for reproducibility
		- replace_val: 0 or np.nan for example
		"""
		np.random.seed(seed)
  
		missing_mask_train = np.random.choice([0, 1], size=self.X_train.shape, p=[1-fraction, fraction])
		missing_mask_valid = np.random.choice([0, 1], size=self.X_valid.shape, p=[1-fraction, fraction])
		missing_mask_test = np.random.choice([0, 1], size=self.X_test.shape, p=[1-fraction, fraction])
   
		if include_train:
			self.X_train_missing = self.X_train.copy()
			self.X_train_missing[missing_mask_train==1] = replace_val
  
		self.X_valid_missing = self.X_valid.copy()
		self.X_valid_missing[missing_mask_valid==1] = replace_val

		self.X_test_missing = self.X_test.copy()
		self.X_test_missing[missing_mask_test==1] = replace_val
	
	def gen_selected_datasets(self, feature_importance):
		"""
		Missing data mechanism
		- feature_importance (np.ndarray): importance of each feature
		- fraction (float): percentage of missing values in [0,1]

		Removes the fraction of entire features from the dataset, starting with the least important.
		"""
		feature_removal_dict = {
			'PBMC': 11000,
			'COIL20': 800,
			'USPS': 200,
			'Isolet': 400,
			'madelon': 480,
			'mice_protein': 60,
			'finance': 100
		}
		# Calculate the threshold to select features with low importance
		sorted_indices = np.argsort(feature_importance)  # sort features by importance
		num_features_to_remove = feature_removal_dict[self.args.dataset]
		features_to_remove = sorted_indices[:num_features_to_remove]  # select the least important features

		# Apply changes to each dataset
		for dataset_name in ['train', 'valid', 'test']:

			attr_name = f'X_{dataset_name}_selected'
			if hasattr(self, attr_name):
				delattr(self, attr_name)

			dataset = getattr(self, f'X_{dataset_name}')
			dataset_missing = np.delete(dataset, features_to_remove, axis=1)  # Remove entire columns

			setattr(self, attr_name, dataset_missing)

		removed_features_mask = np.zeros_like(feature_importance, dtype=int)
		removed_features_mask[features_to_remove] = 1  # Mark the indices in features_to_remove

		return torch.Tensor(removed_features_mask)

	def gen_MNAR_datasets(self, feature_importance, fraction=0.1, replace_val=0, include_train=False):
		"""
		Missing data mechanism
		- feature_importance (np.ndarray): importance of each feature
		- fraction (float): percentage of missing values in [0,1]
		- replace_val (int or float): value used to replace missing entries, e.g., 0 or np.nan

		Removes the fraction of entire features from the dataset, starting with the least important,
		and replaces those entire feature columns with replace_val.
		"""
		# Calculate the threshold to select features with low importance
		sorted_indices = np.argsort(feature_importance)  # sort features by importance
		num_features_to_remove = int(np.ceil(fraction * len(feature_importance)))
		features_to_remove = sorted_indices[:num_features_to_remove]  # select the least important features

		# Apply changes to each dataset
		for dataset_name in ['train', 'valid', 'test']:
			if not include_train:
				if dataset_name == 'train':
					continue
			attr_name = f'X_{dataset_name}_missing'
			if hasattr(self, attr_name):
				delattr(self, attr_name)

			dataset = getattr(self, f'X_{dataset_name}')
			dataset_missing = dataset.copy()
			dataset_missing[:, features_to_remove] = replace_val  # Set entire columns to replace_val

			setattr(self, attr_name, dataset_missing)
   
			removed_features_mask = np.zeros_like(feature_importance, dtype=int)

			# Set the indices in features_to_remove to 1
			removed_features_mask[features_to_remove] = 1

		return torch.Tensor(removed_features_mask)

	def missing_dataloader(self, X_missing, y):
		missing_dataset = CustomPytorchDataset(X_missing, y)

		return DataLoader(missing_dataset, batch_size=self.args.batch_size,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)

	def missing_val_dataloader(self):
		self.missing_valid_dataset = CustomPytorchDataset(self.X_valid_missing, self.y_valid)

		return DataLoader(self.missing_valid_dataset, batch_size=self.args.batch_size,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)

	def missing_test_dataloader(self):
		self.missing_test_dataset = CustomPytorchDataset(self.X_test_missing, self.y_test)

		return DataLoader(self.missing_test_dataset, batch_size=self.args.batch_size,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, persistent_workers=self.args.persistent_workers)


def create_data_module(args):
	if "__" in args.dataset:	# used when instantiang the model from wandb artifacts
		dataset, dataset_size = args.dataset.split("__")
	else:
		dataset, dataset_size = args.dataset, args.dataset_size

	if dataset in [
			'mice_protein', 'MNIST',
			"COIL20", "gisette", "Isolet", "madelon", "USPS",
			"PBMC", "PBMC_small",
			"finance"]:
		if dataset=='mice_protein':
			X, y = load_mice(args)
		elif dataset in ["COIL20", "gisette", "Isolet", "madelon", "USPS"]:
			X, y = load_ASU_dataset(args, dataset)
		elif dataset == "PBMC":
			X, y = load_PBMC_small(args)
		elif dataset == "PBMC_small":
			X, y = load_PBMC_small(args)
		elif dataset == "finance":
			X, y = load_finance(args)


		data_module = create_datamodule_with_cross_validation(args, X, y)

	else:
		raise Exception(f"Dataset <{dataset}> not supported")

	#### Compute classification loss weights
	if args.class_weight=='balanced':
		args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data_module.y_train), y=data_module.y_train)
	elif args.class_weight=='standard':
		args.class_weights = compute_class_weight(class_weight=None, classes=np.unique(data_module.y_train), y=data_module.y_train)
	args.class_weights = args.class_weights.astype(np.float32)
	print(f"Weights for the classification loss: {args.class_weights}")

	return data_module


def create_datamodule_with_cross_validation(args, X, y):
	"""
	Split X, y to be suitable for nested cross-validation.
	It uses args.valid_split and args.test_split to create 
		the train, valid and test stratified datasets.
	"""
	if type(X)==pd.DataFrame:
		X = X.to_numpy()
	if type(y)==pd.Series or type(y)==pd.DataFrame:
		y = y.to_numpy()
	
	if args.dataset_feature_set=='8000':
		X = X[:, :8000]
	elif args.dataset_feature_set=='16000':
		X = X[:, :16000]

	assert type(X)==np.ndarray
	assert type(y)==np.ndarray

	X_train_and_valid, X_test, y_train_and_valid, y_test = compute_stratified_splits(
		X, y, cv_folds=args.cv_folds, seed_kfold=args.seed_kfold, split_id=args.test_split)

	# Split validation set
	X_train, X_valid, y_train, y_valid = train_test_split(
		X_train_and_valid, y_train_and_valid,
		test_size = args.valid_percentage,
		random_state = args.seed_validation,
		stratify = y_train_and_valid
	)
	
	print(f"Train size: {X_train.shape[0]}\n")
	print(f"Valid size: {X_valid.shape[0]}\n")
	print(f"Test size: {X_test.shape[0]}\n")

	assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == X.shape[0]
	assert set(y_train).union(set(y_valid)).union(set(y_test)) == set(y)

	if args.train_on_full_data:
		# Train on the entire training set (train + validation)
		# Validation and test sets are the same
		return DatasetModule(args, X_train_and_valid, y_train_and_valid, X_test, y_test, X_test, y_test)
	else:
		return DatasetModule(args, X_train, y_train, X_valid, y_valid, X_test, y_test)
	

def create_datamodule_with_fixed_test(args, train_path, test_path):
	"""
	Data module suitable when all the splits are pre-made and ready to load from their path.    
	By **convention**, the label is on the last column.
	"""
	assert args.valid_split < args.cv_folds
	assert test_path!=None

	# Load data. By convention the last column is the target
	X_test, y_test = load_csv_data(test_path, labels_column=-1)
	X_train_and_valid, y_train_and_valid = load_csv_data(train_path, labels_column=-1)
	
	# Make CV splits
	X_train, X_valid, y_train, y_valid = compute_stratified_splits(
		args, X_train_and_valid, y_train_and_valid, split_id=args.valid_split)

	assert X_train.shape[0] + X_valid.shape[0] == X_train_and_valid.shape[0]
	assert set(y_train).union(set(y_valid)) == set(y_train_and_valid) 

	return DatasetModule(args, X_train, y_train, X_valid, y_valid, X_test, y_test)