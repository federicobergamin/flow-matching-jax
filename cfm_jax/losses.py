import jax
import optax
import matplotlib.pyplot as plt
# model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

# def cfm_loss(params, x_t, t, ut):
#     # compute predictions
#     vt = model_apply(params, x_t, t)

#     # now i can compute the loss
#     loss = (vt - ut) ** 2
#     return loss.mean()


# def cfm_loss(params, apply_fun, x_t, t, ut):
#     # compute predictions
#     # vt = model.apply({"params": params}, x_t, t)
#     vt = apply_fun(params, x_t, t)

#     # now i can compute the loss
#     loss = (vt - ut) ** 2
#     return loss.mean()


def cfm_loss(apply_fun):
    # compute predictions
    # vt = model.apply({"params": params}, x_t, t)
    def loss(params, x_t, t, ut):
        vt = apply_fun(params, x_t, t)

        # now i can compute the loss
        assert vt.shape == ut.shape, f"Shapes do not match: {vt.shape} and {ut.shape}"
        loss = (vt - ut) ** 2
        return loss.mean()

    return loss


def cross_entropy_loss(apply_fun):
    def loss(params, x, y):
        logits = apply_fun(params, x)

        # now i can compute the loss
        loss =  optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
        return loss.sum(), logits

    return loss

# def cross_entropy_loss2(apply_fun):
#     def loss(params, x, y):
#         logits = apply_fun(params, x)

#         # now i can compute the loss
#         loss =  optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
#         return loss.sum(), logits

#     return loss

def guided_loss(x0, mask):
    def loss(generated, true, t):
        # I apply the mask to both generated and true
        t = t.reshape(-1, *([1] * (len(generated.shape) - 1))) 
        approx_at_time1 = (generated - (1-t) * x0)/t
        masked_approx_at_time1 = approx_at_time1 
        masked_true = true 
        l2_norm = (masked_approx_at_time1 - masked_true) ** 2
        l2_norm = l2_norm * mask
        return l2_norm.mean(), approx_at_time1
    return loss

