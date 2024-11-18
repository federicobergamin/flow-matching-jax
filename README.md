## Conditional Flow Matching in JAX

Minimal and simple implementation of Conditional Flow Matching in JAX. The purpose was to learn about CFM and at the same time play with JAX. Therefore, if there are better way to implement stuff feel free to suggest. It still working in progress especially the conditional sampling for inverse problem part.

To make it work you first have to run the following command `pip install -e .`

Implemented methods:
- "Flow Matching for Generative Modeling" Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). `method=="CFM"`
- Allow to use any p(x0) and p(x1) instead of having p(x0) fixed to N(0,1) (or just rectified flow with 1 iteration)  `method=="CFMv2"`. If p(x0) is N(0,1) then this is equivalent to the previous method.
- "Improving and generalizing flow-based generative models with minibatch optimal transport." Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023). `method=="OT-CFM"`


Highly inspired by the following (and way better) repository:
- [https://github.com/atong01/conditional-flow-matching/tree/main](https://github.com/atong01/conditional-flow-matching/tree/main)
- Mask example inspired by [https://github.com/helibenhamu/GeMSS_flow_matching/](https://github.com/helibenhamu/GeMSS_flow_matching/)


We have several 2D experiments and also an example of training a flow matching model on MNIST. In `experiments/unconditional_and_conditional_sampling_mnist/` you can find two examples to perform conditional generation using classifier guidance and also the code to perform inverse problems (just infilling for now) using an unconditional flow matching model. Note, that for the inverse problems we are not using the weighting factor proposed in the 'Training-free Linear Image Inverses via Flows" but a simple one. 

### Examples

**Conditional generation**

![readme_images/cfm_mnist_samples_reconstruction_guidance_model_7.png](readme_images/cfm_mnist_samples_reconstruction_guidance_model_7.png)

![readme_images/cfm_mnist_samples_reconstruction_guidance_model_5.png](readme_images/cfm_mnist_samples_reconstruction_guidance_model_5.png)

![readme_images/cfm_mnist_samples_reconstruction_guidance_model_8.png](readme_images/cfm_mnist_samples_reconstruction_guidance_model_8.png)

**Inverse problems**

![readme_images/inverse_problems_tasks.png](readme_images/inverse_problems_tasks.png)

### TODO (Priority list):
- imrove classifier we are using for classifier guidance
- improve inverse problems 
- likelihood computation

POSSIBLE EXTENSIONS:
- D flow?
- discrete data?