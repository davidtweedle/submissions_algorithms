"""
Draft submission for a preconditioner which utilizes a randomized svd to compute preconditioned updates.
Before the preconditioner is applied, bias and weight terms are combined and reshaped to a matrix.
If the gradient is G, we first sketch G, then compute the SVD of the result, G ~ USV^T.
(In fact, here we sketch the momentum and compute the update based on the momentum).
Then we update the weights by UV^T.
If G is an m-by-n matrix, and d is the sketching dimension, this costs O(mnd).
The sketching and svd step can be sped up by sketching from the left and right, but the cost of reconstruction will still be O(mnd).
In the case where communication savings will be high enough, U, V can be computed in O(md^2) and O(nd^2) and then all reduced (with lower memory communication costs) and reconstructed after.
LayerNorm and BatchNorm parameters are optimized with nadamw.
"""

import functools
import math
from typing import (
        Any,
        Callable,
        Dict,
        Iterator,
        List,
        NamedTuple,
        Optional,
        Tuple,
        Union,
        )

import chex
import jax
import jax.numpy as jnp
import optax

from flax import struct
from lra_opt import low_rank_orthogonal_update, create_param_labels

from algoperf import spec, jax_sharding_utils

import os, atexit
import jax.profiler as jprof

PROFILE_DIR = "/workspace/logs/jax_profile"
PROFILE_START_STEP = int(os.environ.get("PROFILE_START_STEP", "100"))

_profile_started = False

def _maybe_start_profile(step: int):
    global _profile_started
    if (not _profile_started) and (step >= PROFILE_START_STEP):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        jprof.start_trace(PROFILE_DIR)
        _profile_started = True
        atexit.register(lambda: jprof.stop_trace() if _profile_started else None)

def _maybe_stop_profile():
    global _profile_started
    if _profile_started:
        jprof.stop_trace()
        _profile_started = False


# parameter names -- 'scale','bias' -- LayerNorm/BatchNorm (Apply nadamw)
# 'kernel', 'embedding_table', 'embedding' -- higher-dimensional tensors to apply low rank orthogonal updates

HPARAMS = {
        'beta1': 0.9,           # momentum parameter for orthogonal updates and nadamw
        'beta2': 0.999,         # parameter for nadamw only (for the second moment)
        'krylov_iter': 2,       # number of iterations to use for finding range of input to svd
        'learning_rate': 0.001,  # learning rate
        'eps': 1e-8,            # eps value for nadamw 
        'eps_root': 0.0,        # sqrt(eps) value for nadamw
        'weight_decay': 0.01,   # weight_decay
        'dropout_rate': 0.1,    # dropout
        'rank_type': 'sqrt',    # or 'constant', decides what dimension to sketch
        'rank': None        # if 'rank_type'='constant', then what rank to use
        }

_GRAD_CLIP_EPS = 1e-6


def train_step(workload,
             opt_update_fn,
             model_state,
             optimizer_state,
             current_param_container,
             batch,
             rng,
             grad_clip,
             label_smoothing,
             dropout_rate,
             ):

    def _loss_fn(params):
        logits, new_model_state = workload.model_fn(
                params,
                batch,
                model_state,
                spec.ForwardPassMode.TRAIN,
                rng,
                update_batch_norm=True,
                dropout_rate=dropout_rate,
                )
        loss_dict = workload.loss_fn(
                label_batch=batch['targets'],
                logits_batch=logits,
                mask_batch=batch.get('weights'),
                label_smoothing=label_smoothing)
        summed_loss = loss_dict['summed']
        n_valid_examples = loss_dict['n_valid_examples']
        return summed_loss, (n_valid_examples, new_model_state)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
          current_param_container)
    summed_loss = jax.lax.psum(summed_loss, axis_name='batch')
    total_n_valid_examples = jax.lax.psum(n_valid_examples, axis_name='batch')
    grad = jax.lax.pmean(grad, axis_name='batch')
    # Get correct global mean loss and grad.
    loss = summed_loss / total_n_valid_examples
    grad = jax.tree.map(lambda x: x / total_n_valid_examples, grad)
    grad = jax.lax.pmean(grad, axis_name='batch')

    grad_norm = jnp.sqrt(
          sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

    grad_scaling_factor = 1.0
    if grad_clip is not None:
        grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree.map(lambda x: x * grad_scaling_factor, grad)

    updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
    updated_params = optax.apply_updates(current_param_container, updates)
    return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        batch: Dict[str, spec.Tensor],
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        train_state: Optional[Dict[str, Any]] = None) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types
    del loss_type
    del train_state
    del eval_results
    del hyperparameters

    hyperparameters = HPARAMS

    optimizer_state, opt_update_fn = optimizer_state
    if 'label_smoothing' in hyperparameters:
        label_smoothing = hyperparameters['label_smoothing']
    else:
        label_smoothing = 0.0
    if 'grad_clip' in hyperparameters:
        grad_clip = hyperparameters['grad_clip']
    else:
        grad_clip = None
    dropout_rate = hyperparameters['dropout_rate']

    # mesh = jax.sharding.Mesh(jax.devices(), ('batch'))
    replicated = jax_sharding_utils.get_replicate_sharding()
    sharded = (
            jax_sharding_utils.get_batch_dim_sharding()
            )


    arg_shardings = (
            replicated, #model_state
            replicated, #optimizer_state # change to optimizer sharding eventually
            replicated, # current_param_container
            sharded, # batch
            replicated, # per_device_rngs
            replicated, # grad_clip
            replicated, #label_smoothing
            replicated, #dropout_rate
            )
    out_shardings = (
            replicated, # new_optimizer_state # maybe sharded eventually
            replicated, # updated_params
            replicated, # new_model_state
            replicated, # loss
            replicated, # grad_norm
            )
    jitted_train_step = jax.jit(
            train_step,
            static_argnums=(0, 1),
            donate_argnums=(2, 3, 4),
            in_shardings=arg_shardings,
            out_shardings=out_shardings,
            )

    _maybe_start_profile(global_step)

    outputs = jitted_train_step(workload,
                                opt_update_fn,
                                model_state,
                                optimizer_state,
                                current_param_container,
                                batch,
                                rng,
                                grad_clip,
                                label_smoothing,
                                dropout_rate,
                                )
    new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

    # Log loss, grad_norm.
    if global_step % 100 == 0 and workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
                {
                    'loss': loss,
                    'grad_norm': grad_norm,
                    }, global_step)
    return (new_optimizer_state, opt_update_fn), new_params, new_model_state



def prepare_for_eval(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        ) -> spec.UpdateReturn:
    del workload
    del hyperparameters
    del current_params_types
    del loss_type
    del eval_results
    del global_step
    del rng
    _maybe_stop_profile()
    return (optimizer_state, current_param_container, model_state)

def get_batch_size(workload_name):
    if workload_name == 'criteo1tb':
        return 262_144
    elif workload_name == 'fastmri':
        return 32
    elif workload_name == 'imagenet_resnet':
        return 1024
    elif workload_name == 'imagenet_resnet_silu':
        return 512
    elif workload_name == 'imagenet_resnet_gelu':
        return 512
    elif workload_name == 'imagenet_vit':
        return 1024
    elif workload_name == 'librispeech_conformer':
        return 256
    elif workload_name == 'librispeech_deepspeech':
        return 256
    elif workload_name == 'ogbg':
        return 512
    elif workload_name == 'wmt':
        return 128
    elif workload_name == 'mnist':
        return 16
    elif workload_name == 'cifar':
        return 1024
    else:
        raise ValueError(f'Unsupported workload name: {workload_name}.')

def data_selection(
        workload: spec.Workload,
        input_queue: Iterator[Dict[str, spec.Tensor]],
        optimizer_state: spec.OptimizerState,
        current_param_container: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        global_step: int,
        rng: spec.RandomState,
        ) -> Dict[str, spec.Tensor]:
    del workload
    del optimizer_state
    del current_param_container
    del model_state
    del hyperparameters
    del global_step
    del rng
    batch = next(input_queue)
    return batch

def init_optimizer_state(
        workload: spec.Workload,
        model_params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        rng: spec.RandomState,
        ) -> spec.OptimizerState:
    del model_params
    del model_state
    params_zeros_like = jax.tree.map(
            lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes
            )
    lr = HPARAMS['learning_rate']
    beta1 = HPARAMS['beta1']
    beta2 = HPARAMS['beta2']
    weight_decay = HPARAMS['weight_decay']
    krylov_iter = HPARAMS['krylov_iter']
    rank_type = HPARAMS['rank_type']  # 'sqrt' or 'constant'
    if rank_type == 'constant':
        rank_val = HPARAMS['rank']
    else:
        rank_val = None

    param_label_fn = create_param_labels()  # (params_zeros_like)


    opt_init_fn, opt_update_fn = low_rank_orthogonal_update(
            key=rng,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            krylov_iter=krylov_iter,
            rank_type=rank_type,
            rank_val=rank_val,
            param_label_fn=param_label_fn
            )
    optimizer_state = opt_init_fn(params_zeros_like)
    return optimizer_state, opt_update_fn
