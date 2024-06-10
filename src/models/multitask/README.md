# Source

This multitask code for the huggingface `Trainer` class is based on this [Github repository](https://github.com/shahrukhx01/multitask-learning-transformers).

# Changes

The ideas found in the project mentioned above were generalized 
and support for creating multitask models was added.
The models can be configured via a `TaskConfig` class.

In addition to that, the provided Trainer classes also allow you to
specify a `loss_weight`, which can be useful if you are working with datasets 
or loss functions which have significant differences in their value ranges.
This was achieved by hooking the `compute_loss` function of the huggingface `Trainer` class.

We also added support for a different `compute_metric` function for each task.
In order to assign each sample in a batch to a task, we hooked the `evaluation_loop` of the `Trainer` class.

Another change to the original project, is that the `Trainer` modified for the shared encoder variant 
is now able to properly save and load models.