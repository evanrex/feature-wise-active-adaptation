# create the GP layer called after the neural network
# using **one** GP per feature (as in the SV-DKL paper)
### the outputs of these GPs will be mixed in the softmax likelihood
from json import encoder
from math import e, gamma
from modulefinder import STORE_OPS
from grpc import xds_channel_credentials
from importlib_metadata import version
from pyro import param, sample
import scipy
from sklearn.utils import axis0_safe_slice
from torch.nn.functional import embedding
import wandb
from _shared_imports import *
from torch import nn

from lookahead_optimizer import Lookahead
from sparsity import LearnableSparsityVector, SparsityNetwork

def get_labels_lists(outputs):
	all_y_true, all_y_pred = [], []
	for output in outputs:
		all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
		all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

	return all_y_true, all_y_pred


def compute_all_metrics(args, y_true, y_pred):
	metrics = {}
	metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
	metrics['F1_weighted'] = f1_score(y_true, y_pred, average='weighted')
	metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
	metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
	if args.num_classes==2:
		metrics['AUROC_weighted'] = roc_auc_score(y_true, y_pred, average='weighted')
	
	return metrics


def detach_tensors(tensors):
	"""
	Detach losses 
	"""
	if type(tensors)==list:
		detached_tensors = list()
		for tensor in tensors:
			detach_tensors.append(tensor.detach())
	elif type(tensors)==dict:
		detached_tensors = dict()
		for key, tensor in tensors.items():
			detached_tensors[key] = tensor.detach()
	else:
		raise Exception("tensors must be a list or a dict")
	
	return detached_tensors

def reshape_batch(batch):
	"""
	When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
	
	This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
	"""
	x, y = batch
	x = x.reshape(-1, x.shape[-1])
	y = y.reshape(-1)

	return x, y


def create_model(args, data_module):
	"""
	Returns a model instance.
	"""
	pl.seed_everything(args.seed_model_init, workers=True)
 
	if args.model == 'fwal':
		model = FWAL(args)
	
	elif args.model=='cae': 
		model = CAE(args)

	else:
		raise Exception(f"The model ${args.model}$ is not supported")

	return model


class ConcreteLayer(nn.Module):
	"""
	Implementation of a concrete layer from paper "Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"
	"""

	def __init__(self, args, input_dim, output_dim, is_diet_layer=False, wpn_embedding_matrix=None):
		"""
		- input_dim (int): dimension of the input
		- output_dim (int): number of neurons in the layer
		"""
		super().__init__()
		self.args = args
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.temp_start = 10
		self.temp_end = 0.01
		# the iteration is used in annealing the temperature
		# 	it's increased with every call to sample during training
		self.current_iteration = 0 
		self.anneal_iterations = args.concrete_anneal_iterations # maximum number of iterations for the temperature optimization

		self.alphas = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=True)
		torch.nn.init.xavier_normal_(self.alphas, gain=1) # Glorot normalization, following the original CAE implementation
		
	def get_temperature(self):
		# compute temperature		
		if self.current_iteration >= self.anneal_iterations:
			return self.temp_end
		else:
			return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)

	def sample(self):
		"""
		Sample from the concrete distribution.
		"""
		# Increase the iteration counter during training
		if self.training:
			self.current_iteration += 1

		temperature = self.get_temperature()

		alphas = self.alphas # alphas is a K x D matrix

		# sample from the concrete distribution
		if self.training:
			samples = F.gumbel_softmax(alphas, tau=temperature, hard=False) # size K x D
			assert samples.shape == (self.output_dim, self.input_dim)
		else: 			# sample using argmax
			index_max_alphas = torch.argmax(alphas, dim=1) # size K
			samples = torch.zeros(self.output_dim, self.input_dim).cuda()
			samples[torch.arange(self.output_dim), index_max_alphas] = 1.

		return samples

	def forward(self, x):
		"""
		- x (batch_size x input_dim)
		"""
		mask = self.sample()   	# size (number_neurons x input_dim)
		x = torch.matmul(x, mask.T) 		# size (batch_size, number_neurons)
		return x, None # return additional None for compatibility



""""
Metrics
- all 
	- balanced_accuracy
	- F1 - weighted
	- precision - weighted
	- recall - weighted
	- accuracy per class
- binary
	- AUROC (binary)
- loss
	- total
	- reconstruction
	- cross-entropy
"""

class TrainingLightningModule(pl.LightningModule):
	"""
	General class to be inherited by all implemented models (e.g., MLP, CAE, FsNet etc.)

	It implements general training and evaluation functions (e.g., computing losses, logging, training etc.)
	"""
	def __init__(self, args):
		super().__init__()
		self.training_step_outputs = []
		self.validation_step_outputs = []
		self.test_step_outputs = []
		self.args = args
		self.learning_rate = args.lr
  
		if args.reconstruction_loss == "mse":
			self.reconstruction_loss = F.mse_loss
		elif args.reconstruction_loss == "bce":
			self.reconstruction_loss = F.binary_cross_entropy

	def compute_loss(self, y_true, y_hat, x, x_hat, sparsity_weights):
		losses = {}
		losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))
		
		if x_hat is None:
			losses['reconstruction'] = torch.tensor(0., device=self.device)
		else:
			losses['reconstruction'] = self.args.gamma * self.reconstruction_loss(x_hat, x, reduction='sum') #if self.decoder else torch.zeros(1, device=self.device)
  
		if sparsity_weights is None:
			losses['sparsity'] = torch.tensor(0., device=self.device)
			
		else:
			if self.args.sparsity_regularizer=='L1':
				losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * torch.norm(sparsity_weights, 1)
			elif self.args.sparsity_regularizer=='hoyer':
				hoyer_reg = torch.norm(sparsity_weights, 1) / torch.norm(sparsity_weights, 2)
				losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * hoyer_reg
			else:
				raise Exception("Sparsity regularizer not valid")

			if self.args.normalize_sparsity:
				losses['sparsity'] = (1/len(sparsity_weights)) * losses['sparsity']

		if self.args.normalize_reconstruction:
			if self.args.normalize_reconstruction == 'num_features':
				losses['reconstruction'] = (1/len(sparsity_weights)) * losses['reconstruction']
			elif self.args.normalize_reconstruction == 'num_non_masked_features':
				losses['reconstruction'] = (1/torch.sum(sparsity_weights)) * losses['reconstruction']
			else:
				raise Exception(f"Normalization method  <{self.args.normalize_reconstruction}> not valid. Must be one of ['num_features', 'num_non_masked_features']")
		
		if self.args.as_MLP_baseline:
			losses['sparsity'] = torch.tensor(0., device=self.device)
			losses['reconstruction'] = torch.tensor(0., device=self.device)
		


		losses['total'] = losses['cross_entropy'] + losses['reconstruction'] + losses['sparsity']
		
		return losses
	
    ### DEFINE SELF-SUPERVISED LOSS FUNCTION
	def pre_loss(self, x_true, x_pred, mask_true, mask_pred):
		losses = {}
		losses['pre_reconstruction'] = self.reconstruction_loss(x_pred, x_true, reduction='mean')
  
		if mask_true is None or mask_pred is None:
			losses['pre_cross_entropy'] = torch.tensor(0., device=self.device)
		else:
			losses['pre_cross_entropy'] = F.binary_cross_entropy_with_logits(mask_pred, mask_true)
        
		losses['pre_total'] = losses['pre_reconstruction'] + self.args.pre_alpha * losses['pre_cross_entropy']
		return losses

	def log_pre_losses(self, losses, key, dataloader_name=""):
		self.log(f"{key}/pre_total_loss{dataloader_name}", losses['pre_total'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/pre_reconstruction_loss{dataloader_name}", losses['pre_reconstruction'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/pre_cross_entropy_loss{dataloader_name}", losses['pre_cross_entropy'].item(), sync_dist=self.args.hpc_run)

	def log_losses(self, losses, key, dataloader_name=""):
		self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/reconstruction_loss{dataloader_name}", losses['reconstruction'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/sparsity_loss{dataloader_name}", losses['sparsity'].item(), sync_dist=self.args.hpc_run)

	def log_epoch_metrics(self, outputs, key, dataloader_name=""):
		y_true, y_pred = get_labels_lists(outputs)
		if not self.args.pretrain:
			self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred), sync_dist=self.args.hpc_run)
		self.log(f'{key}/F1_weighted{dataloader_name}', f1_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		self.log(f'{key}/precision_weighted{dataloader_name}', precision_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		self.log(f'{key}/recall_weighted{dataloader_name}', recall_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		if self.args.num_classes==2:
			self.log(f'{key}/AUROC_weighted{dataloader_name}', roc_auc_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)

	def pre_training_step(self, batch, batch_idx):
		x, y_true = batch
	
		x_pred, mask, mask_pred = self.pre_forward(x)

		losses = self.pre_loss(x, x_pred, mask, mask_pred)

		self.log_pre_losses(losses, key='pre_train')

		
		outputs = {
			'loss': losses['pre_total'],
			'losses': detach_tensors(losses),
			'y_true': mask,
			'y_pred': (mask_pred>0).float()
		}
		self.training_step_outputs.append(outputs)
		return outputs
	
	def training_step(self, batch, batch_idx):
		x, y_true = batch

		if self.args.pretrain:
			return self.pre_training_step(batch, batch_idx)
	
		y_hat, x_hat, sparsity_weights = self.forward(x)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		self.log_losses(losses, key='train')
		# self.log("train/lr", self.learning_rate)
		
		# log temperature of the concrete distribution
		# if isinstance(self.first_layer, ConcreteLayer):
		# 	self.log("train/concrete_temperature", self.first_layer.get_temperature())
		if self.args.model=='fwal' and self.args.mask_type=='gumbel_softmax':
			self.log("train/gumbel_temperature", self.get_temperature())
		outputs = {
			'loss': losses['total'],
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}
		self.training_step_outputs.append(outputs)
		return outputs

	def on_train_epoch_end(self):
		if self.args.pretrain:
			self.log_epoch_metrics(self.training_step_outputs, 'pre_train')
		else:
			self.log_epoch_metrics(self.training_step_outputs, 'train')
		self.training_step_outputs.clear()  # free memory

	def pre_validation_step(self, batch, batch_idx, dataloader_idx=0):
		x, y_true = reshape_batch(batch)

		x_pred, mask, mask_pred = self.pre_forward(x)

		losses = self.pre_loss(x, x_pred, mask, mask_pred)

		self.log_pre_losses(losses, key='pre_valid')
		
		output = {
			'losses': detach_tensors(losses),
			'y_true': mask,
			'y_pred': (mask_pred>0).float()
		}
		
		while len(self.validation_step_outputs) <= dataloader_idx:
			self.validation_step_outputs.append([])
   
		self.validation_step_outputs[dataloader_idx].append(output)
		return output

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		"""
		- dataloader_idx (int) tells which dataloader is the `batch` coming from
		"""

		if self.args.pretrain:
			return self.pre_validation_step(batch, batch_idx, dataloader_idx)
		
		x, y_true = reshape_batch(batch)


		y_hat, x_hat, sparsity_weights = self.forward(x, test_time=True)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		output = {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}
		while len(self.validation_step_outputs) <= dataloader_idx:
			self.validation_step_outputs.append([])
   
		self.validation_step_outputs[dataloader_idx].append(output)
		return output

	def on_validation_epoch_end(self):
		"""
		- outputs: when no_dataloaders==1 --> A list of dictionaries corresponding to a validation step.
				   when no_dataloaders>1  --> List with length equal to the number of validation dataloaders. Each element is a list with the dictionaries corresponding to a validation step.
		"""
		### Log losses and metrics
		# `outputs_all_dataloaders` is expected to a list of dataloaders.
		# However, when there's only one dataloader, outputs_all_dataloaders is NOT a list.
		# Thus, we transform it in a list to preserve compatibility
		outputs_all_dataloaders = self.validation_step_outputs

		for dataloader_id, outputs in enumerate(outputs_all_dataloaders):
			
			if self.args.pretrain:
				
				losses = {
					'pre_total': np.mean([output['losses']['pre_total'].item() for output in outputs]),
					'pre_reconstruction': np.mean([output['losses']['pre_reconstruction'].item() for output in outputs]),
					'pre_cross_entropy': np.mean([output['losses']['pre_cross_entropy'].item() for output in outputs]),
				}
				if dataloader_id==0: # original validation dataset
					dataloader_name=""
				else:
					dataloader_name=f"__{self.args.val_dataloaders_name[dataloader_id]}"

				self.log_pre_losses(losses, key='pre_valid', dataloader_name=dataloader_name)
				self.log_epoch_metrics(outputs, key='pre_valid', dataloader_name=dataloader_name)
			else:

				losses = {
					'total': np.mean([output['losses']['total'].item() for output in outputs]),
					'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
					'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
					'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
				}
				if dataloader_id==0: # original validation dataset
					dataloader_name=""
				else:
					dataloader_name=f"__{self.args.val_dataloaders_name[dataloader_id]}"

				self.log_losses(losses, key='valid', dataloader_name=dataloader_name)
				self.log_epoch_metrics(outputs, key='valid', dataloader_name=dataloader_name)

				if self.args.model =='fwal':
					self.masks.append(self.mask.data.cpu().numpy().copy())
     
		self.validation_step_outputs.clear()

	def test_step(self, batch, batch_idx, dataloader_idx=0):
		'''accommodates multiple dataloaders'''
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x, test_time=True)
		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		output =  {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1),
			'y_hat': y_hat.detach().cpu().numpy()
		}
		while len(self.test_step_outputs) <= dataloader_idx:
			self.test_step_outputs.append([])
   
		self.test_step_outputs[dataloader_idx].append(output)
  
		return output

	def on_test_epoch_end(self):
		'''accommodates multiple dataloaders but only uses first'''

		outputs = self.test_step_outputs[0]

		### Save losses
		losses = {
			'total': np.mean([output['losses']['total'].item() for output in outputs]),
			'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
			'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
			'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
		}
		self.log_losses(losses, key=self.log_test_key)
		self.log_epoch_metrics(outputs, self.log_test_key)

		#### Save prediction probabilities
		y_hat_list = [output['y_hat'] for output in outputs]
		y_hat_all = np.concatenate(y_hat_list, axis=0)
		y_hat_all = scipy.special.softmax(y_hat_all, axis=1)

		y_hat_all = wandb.Table(dataframe=pd.DataFrame(y_hat_all))
		wandb.log({f'{self.log_test_key}_y_hat': y_hat_all})


		### Save global feature importances
		# if self.args.sparsity_type == 'global':
		# 	feature_importance = self.feature_extractor.sparsity_model.forward(None).cpu().detach().numpy()
			
		# 	global_feature_importance = wandb.Table(dataframe=pd.DataFrame(feature_importance))
		# 	wandb.log({f'{self.log_test_key}_global_feature_importance': global_feature_importance})
	
	def configure_optimizers(self):
		params = self.parameters()

		if self.args.optimizer=='adam':
			optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
		if self.args.optimizer=='adamw':
			optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])
		
		if self.args.lookahead_optimizer:
			optimizer = Lookahead(optimizer, la_steps=5, la_alpha=0.5)

		if self.args.lr_scheduler == None:
			return optimizer
		else:
			if self.args.lr_scheduler == 'plateau':
				lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
			elif self.args.lr_scheduler == 'cosine_warm_restart':
				# Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
				# 	splits the scheduler for 6 periods. We split into 6 periods as well.
				lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
					T_0 = self.args.cosine_warm_restart_t_0,
					eta_min = self.args.cosine_warm_restart_eta_min,
					verbose=True)
			elif self.args.lr_scheduler == 'linear':
				lr_scheduler = torch.optim.lr_scheduler.LinearLR(
					optimizer, 
					start_factor = self.args.lr,
					end_factor = 3e-5,
					total_iters = self.args.max_steps / self.args.val_check_interval)
			elif self.args.lr_scheduler == 'lambda':
				def scheduler(epoch):
					if epoch < 500:
						return 0.995 ** epoch
					else:
						return 0.1

				lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
					optimizer,
					scheduler)
			else:
				raise Exception()

			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': lr_scheduler,
					'monitor': 'valid/cross_entropy_loss',
					'interval': 'step',
					'frequency': self.args.val_check_interval,
					'name': 'lr_scheduler'
				}
			}


class CAE(TrainingLightningModule):
	def __init__(self, args):
		super().__init__(args)
		self.args = args
		self.log_test_key = None
		self.learning_rate = args.lr
		# self.temp_start = 10
		# self.temp_end = 0.01
		# the iteration is used in annealing the temperature
		# 	it's increased with every call to sample during training
		# self.current_iteration = 0 
		# self.anneal_iterations = args.concrete_anneal_iterations 
   
		self.decoder=True
		self.first_layer = None
  
		self.concrete_layer = ConcreteLayer(args, args.num_features, args.num_CAE_neurons)

		# Reconstruction Module: 5 layers with 50 neurons each
		self.reconstruction_module = ReconstructionModule(args, args.num_CAE_neurons, args.num_features)
        
		# Prediction Module: 5 layers with 50 neurons each
		self.prediction_module = PredictionModule(args, args.num_CAE_neurons, args.num_classes)

	def pre_forward(self, x):
		
		masked_x, _ = self.concrete_layer(x)			   # pass through first layer
		
		reconstructed_x = self.reconstruction_module(masked_x)
		
		return reconstructed_x, None, None
    
	def forward(self, x, test_time=False):
		"""
		Forward pass for training
		"""
		with torch.no_grad():
			masked_x, _ = self.concrete_layer(x)	
				
		prediction = self.prediction_module(masked_x)

		return prediction, None, None
   
	def finish_pretraining(self):
		self.freeze_concrete_layer()

	def freeze_concrete_layer(self):
		for param in self.concrete_layer.parameters():
			param.requires_grad = False
	

class ReconstructionModule(nn.Module):
	def __init__(self, args, in_dim, out_dim):
		super().__init__()
		self.args = args
		self.reconstruction_weights = nn.Sequential(
				nn.Linear(in_dim, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, out_dim)  # Last layer outputs num_features
			)


	def forward(self, x):
		out = self.reconstruction_weights(x)
		if self.args.reconstruction_loss == "bce":
			out = torch.sigmoid(out)
		return out


class PredictionModule(nn.Module):
	def __init__(self, args, in_dim, out_dim):
		super().__init__()
		self.args = args
		self.weights = nn.Sequential(
				nn.Linear(in_dim, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, 50),
				nn.ReLU(),
				nn.Linear(50, out_dim)  # Last layer outputs num_features
			)


	def forward(self, x):
		out = self.weights(x)
		return out


class FWAL(TrainingLightningModule):
	def __init__(self, args):
		super().__init__(args)
		self.args = args
		self.log_test_key = None
		self.learning_rate = args.lr
		self.temp_start = 10
		self.temp_end = 0.01
		# the iteration is used in annealing the temperature
		# 	it's increased with every call to sample during training
		self.current_iteration = 0 
		self.anneal_iterations = args.concrete_anneal_iterations 
		self.masks = []

		if self.args.mask_init_value is not None:
			self.mask = nn.Parameter(torch.full((args.num_features,), args.mask_init_value, dtype=torch.float32), requires_grad=True)  
		elif args.mask_init_p_array is not None:
			probabilities = torch.Tensor([float(x) for x in args.mask_init_p_array.split(',')])
			logits = torch.log(probabilities / (1 - probabilities))
			self.mask = nn.Parameter(logits, requires_grad=True)
		else:
			mask_generator = torch.Generator().manual_seed(args.seed_model_mask)
			self.mask = nn.Parameter(torch.randn(args.num_features, generator=mask_generator), requires_grad=True)

   
		self.decoder=True
		self.first_layer = None

		# Reconstruction Module: 5 layers with 50 neurons each
		self.reconstruction_module = ReconstructionModule(args, in_dim=args.num_features, out_dim=args.num_features)
        
        # define weights for pre-training task
		self.mask_predictor_module = nn.Sequential( 
			nn.Linear(args.num_features, 50),
			nn.ReLU(),
			nn.Linear(50, 50),
			nn.ReLU(),
			nn.Linear(50, 50),
			nn.ReLU(),
			nn.Linear(50, 50),
			nn.ReLU(),
			nn.Linear(50, args.num_features)  # Last layer outputs num_features
		)

		# Prediction Module: 5 layers with 50 neurons each
		self.prediction_module = PredictionModule(args, in_dim=args.num_features, out_dim=args.num_classes)
    
	def mask_module(self, x, test_time=False):
		# constructing sparsity weights from mask module
		if self.args.as_MLP_baseline:
			return x, torch.ones_like(x), torch.ones_like(x)
		if self.args.mask_type == "sigmoid":
			sparsity_weights = torch.sigmoid(self.mask)
		elif self.args.mask_type == "gumbel_softmax":
			if test_time:
				sparsity_weights = (self.mask>0).float()
				sparsity_weights_probs = torch.sigmoid(self.mask)
			else:
				if self.training:
					self.current_iteration += 1
				temperature = self.get_temperature()
				soft_outputs = torch.nn.functional.gumbel_softmax(torch.stack((self.mask,-1*self.mask),dim=1), tau=temperature, hard=False, dim=-1)
				_, max_indices = soft_outputs.max(dim=-1, keepdim=True)
				hard_outputs = torch.zeros_like(soft_outputs).scatter_(-1, max_indices, 1.0)[:,0]
				soft_outputs=soft_outputs[:,0]
				sparsity_weights = hard_outputs - soft_outputs.detach() + soft_outputs
				sparsity_weights_probs = soft_outputs
		else:
			raise NotImplementedError(f"mask_type: <{self.args.mask_type}> is not supported. Choose one of [sigmoid, gumbel_softmax]")
			
		return x * sparsity_weights, sparsity_weights, sparsity_weights_probs

	def get_temperature(self):
		# compute temperature		
		if self.current_iteration >= self.anneal_iterations:
			return self.temp_end
		else:
			return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)
		
	def pre_forward(self, x):
		# Ensure pi is a tensor and has the shape compatible with x's features
		pi_tensor = torch.full(x.shape, self.args.pre_pi, device=x.device, dtype=x.dtype)
		
		# Sample from the multivariate Bernoulli distribution
		mask = torch.bernoulli(pi_tensor)
		
		# Select the features based on the mask
		masked_x = x * mask
		
		reconstructed_x = self.reconstruction_module(masked_x)
		
		mask_pred = self.mask_predictor_module(reconstructed_x)
		
		return reconstructed_x, mask, mask_pred
    
	def forward(self, x, test_time=False):
		"""
		Forward pass for training
		"""

		masked_x, sparsity_weights, sparsity_weights_probs = self.mask_module(x, test_time=test_time)
		
		reconstructed_x = self.reconstruction_module(masked_x)
		
		prediction = self.prediction_module(reconstructed_x)

		reconstructed_x = (1-sparsity_weights)*reconstructed_x + sparsity_weights*masked_x  # only want loss for reconstructed x terms that were masked
		
		return prediction, reconstructed_x, sparsity_weights_probs
    
	def necessary_features(self, k=None):
		"""
		k: (int) Defaults to args.num_necessary features. Specifies the number of desired necessary features.
		Returns a boolean mask for which features are deemed necessary and which are not.
		"""
		
		if self.args.as_MLP_baseline:
			return torch.ones_like(self.mask)
		
		if k == None:
			k = self.args.num_necessary_features
		if self.args.mask_type == "sigmoid":
			sigmoid_mask = torch.sigmoid(self.mask)
			_, indices = torch.topk(sigmoid_mask, k)  # Get the indices of the top k values
			boolean_mask = torch.full_like(sigmoid_mask, False, dtype=torch.bool)  # Create a boolean mask initialized to False
			boolean_mask[indices] = True  # Set the top k indices to True
			return boolean_mask
		elif self.args.mask_type =="gumbel_softmax":
			return self.mask > 0
		else:
			raise NotImplementedError(f"mask_type: <{self.args.mask_type}> is not supported. Choose one of [sigmoid, gumbel_softmax]")

	def finish_pretraining(self):
		self.current_iteration = 0
			
	def inference(self, x):
		"""
		Performs inference with test-time interventions
		"""
		x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x

		# Step 1: Replace None (or equivalent) values with 0
		# Assuming x is a tensor and None values are represented as NaNs
		original_x = x.clone()  # Keep a copy of the original x for later
		x = torch.nan_to_num(x, nan=0.0)
		
		masked_x, sparsity_weights, sparsity_weights_probs = self.mask_module(x, test_time=True)
		
		# Step 2: Reconstruction as in the original forward pass
		reconstructed_x = self.reconstruction_module(masked_x)
		
		# Step 3: Replace reconstructed values with original non-None values
		# Assuming None values were NaN and were replaced with 0 in original_x
		mask_non_none = ~torch.isnan(original_x)  # True for non-None original values
		reconstructed_x = torch.where(mask_non_none, original_x, reconstructed_x)
		
		# Return the reconstructed_x and prediction
		prediction = self.prediction_module(reconstructed_x)
		
		prediction = torch.softmax(prediction, dim=1)
		
		return prediction

