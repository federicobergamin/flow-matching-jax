"""
Training a classifier on noisy MNIST digits. 
We need this for the reconstruction guidance experiments.
"""

import jax
import jax.numpy as jnp
from jax import flatten_util

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import optax


from cfm_jax.models.models import ConvNet
from cfm_jax.conditional_flow_matching import ConditionalFlowMatchingModel
from cfm_jax.losses import cross_entropy_loss
from cfm_jax.datautils import sample_gaussian, sample_8gaussians
from cfm_jax.utils import (
    split_key,
    numpy_collate,
    image_to_numpy_zero_one_interval,
)

import pickle as pkl


def main():
    seed = 0
    batch_size = 256
    num_classes = 10
    channels = [8, 16]
    kernel_size = 3
    epochs = 30
    train_for_different_noise_levels = True
    method = "CFMv2"
    saving_path = "trained_models/"

    dataset_path = "data/"

    pnrg_key = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    # jax.random.seed(seed)

    # load the data
    # here I have to split the data into train and validation
    train_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=True, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                   generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    test_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    # generate the model
    model = ConvNet(channels=channels, num_classes=num_classes, act_fn=jax.nn.relu, kernel_size=kernel_size)

    pnrg_key, data_key, init_key = split_key(pnrg_key, num=3)
    
    # dummy input
    x = jax.random.normal(pnrg_key, (batch_size, 28, 28, 1))
    print("initializing the model")
    params_dict = model.init(init_key, x)

    params_vec, unflatten = flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    lr = 0.001
    optim = optax.adam(lr)
    opt_state = optim.init(params_dict["params"])

    # function to apply the model
    model_apply = lambda p, x: model.apply({"params": p}, x)

    # loss function definition
    ce_loss = cross_entropy_loss(model_apply)
    loss_grad_fn = jax.jit(jax.value_and_grad(ce_loss, argnums=0, has_aux=True))

    if train_for_different_noise_levels:
        # now I have to create the CFM model
            flow_matching_model = ConditionalFlowMatchingModel(sigma=0.0, method=method)


    best_valid_accuracy = 0
    for ep in range(epochs):
        ep_loss = 0
        ep_train_accuracy = 0
        ep_examples = 0
        for i, (batch_data, batch_labels) in enumerate(train_loader):

            if train_for_different_noise_levels:
                # now I have to add noise as it is done in the conditional flow matching paper
                key_t, key_p0, key_eps, pnrg_key = split_key(pnrg_key, num=4)
                t = jax.random.uniform(key_t, shape=(batch_data.shape[0],))

                # now I have to sample form p0, which in this case is a single Gaussian
                # distribution
                batch_p0 = sample_gaussian(batch_data.shape[0], dimension=28 * 28, key=key_p0)
                # I have to reshape them
                batch_p0 = batch_p0.reshape(batch_data.shape[0], 28, 28, 1)

                # and get the noised data
                batch_xt = flow_matching_model.sample_xt(batch_p0, batch_data, t, key_epsilon=key_eps)
                
                (loss, logits), grads = loss_grad_fn(params_dict["params"], batch_xt, batch_labels)

            else:
                (loss, logits), grads = loss_grad_fn(params_dict["params"], batch_data, batch_labels)
            # if i==0:
            #     print(logits.shape)
            #     print(jnp.sum(jax.nn.softmax(logits, axis=1), axis=1))
            #     print(jnp.argmax(logits, axis=1))

            updates, opt_state = optim.update(grads, opt_state)
            params_dict["params"] = optax.apply_updates(params_dict["params"], updates)

            ep_loss += loss * batch_data.shape[0] 
            ep_examples += batch_data.shape[0]
            ep_train_accuracy += jnp.sum(jnp.argmax(logits, axis=1) == batch_labels)


        # validation loop
        ep_val_accuracy = 0
        
        for i, (batch_data, batch_labels) in enumerate(valid_loader):
            if train_for_different_noise_levels:
                key_t, key_p0, key_eps, pnrg_key = split_key(pnrg_key, num=4)
                t = jax.random.uniform(key_t, shape=(batch_data.shape[0],))

                # now I have to sample form p0, which in this case is a single Gaussian
                # distribution
                batch_p0 = sample_gaussian(batch_data.shape[0], dimension=28 * 28, key=key_p0)
                # I have to reshape them
                batch_p0 = batch_p0.reshape(batch_data.shape[0], 28, 28, 1)

                batch_xt = flow_matching_model.sample_xt(batch_p0, batch_data, t, key_epsilon=key_eps)
                
                logits = model_apply(params_dict["params"], batch_xt)
            else:
                logits = model_apply(params_dict["params"], batch_data)

            ep_val_accuracy += jnp.sum(jnp.argmax(logits, axis=1) == batch_labels)

        print(f"Epoch {ep}, train loss: {ep_loss/len(train_loader.dataset)}, train accuracy: {ep_train_accuracy/len(train_loader.dataset)}, valid accuracy: {ep_val_accuracy/len(valid_loader.dataset)}")
        
        if ep_val_accuracy/len(valid_loader.dataset) > best_valid_accuracy:
            best_valid_accuracy = ep_val_accuracy/len(valid_loader.dataset)
            best_params_dict = params_dict.copy()
            # here I store the best model based on validation
            # with open("classification_best_model.pkl", "wb") as f:
            #     pkl.dump(params_dict, f)

    # save the best model
    if train_for_different_noise_levels:
        with open(saving_path+"classification_best_model_noised.pkl", "wb") as f:
            pkl.dump(flow_matching_model, f)
    else:
        with open(saving_path+"classification_best_model.pkl", "wb") as f:
            pkl.dump(best_params_dict, f)

    # at the end of the training, I can load the best model
    # with open("classification_best_model.pkl", "rb") as f:
    #     best_params_dict = pkl.load(f)

    # I can now compute the accuracy on the test set
    test_accuracy = 0
    for i, (batch_data, batch_labels) in enumerate(test_loader):
        logits = model_apply(best_params_dict["params"], batch_data)
        test_accuracy += jnp.sum(jnp.argmax(logits, axis=1) == batch_labels)

    print(f"Test accuracy: {test_accuracy/len(test_loader.dataset)}")

    if train_for_different_noise_levels:
        # I can now compute the accuracy on the noised test set
        test_accuracy = 0
        for i, (batch_data, batch_labels) in enumerate(test_loader):
            key_t, key_p0, key_eps, pnrg_key = split_key(pnrg_key, num=4)
            t = jax.random.uniform(key_t, shape=(batch_data.shape[0],))

            # now I have to sample form p0, which in this case is a single Gaussian
            # distribution
            batch_p0 = sample_gaussian(batch_data.shape[0], dimension=28 * 28, key=key_p0)
            # I have to reshape them
            batch_p0 = batch_p0.reshape(batch_data.shape[0], 28, 28, 1)
            
            batch_xt = flow_matching_model.sample_xt(batch_p0, batch_data, t, key_epsilon=key_eps)
            
            logits = model_apply(best_params_dict["params"], batch_xt)
            test_accuracy += jnp.sum(jnp.argmax(logits, axis=1) == batch_labels)

        print(f"Noised test accuracy: {test_accuracy/len(test_loader.dataset)}")





if __name__ == "__main__":
    main()