#from __future__ import print_function
#import numpy as np
import sys
import os
#from cntk import Trainer, learning_rate_schedule, UnitType
#from cntk.learner import sgd
from cntk.ops import *
#import cntk

#cntk.device.set_default_device(cntk.device.cpu())

input_dim = 1 << 18
num_output_classes = 1

print("dim "+ str(input_dim))

# change anyway
input = input_variable(input_dim, np.float32, name='features', is_sparse=True)

def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))

    # times vs transposetimes
    return times(input_var, weight_param) # + bias_param

# alternative -q ab
# c = splice(a,b)
# square( times_transpose (weights, c))
# non-linearity (e.g. sigmoid)( times_transpose (weights, c)) 
# a*a , b*b, a*b

output_dim = num_output_classes
z = linear_layer(input, output_dim)

z.save("simple_cntk_model.cntk")

# label = input_variable((num_output_classes), np.float32)
# loss = cross_entropy_with_softmax(z, label)
# eval_error = classification_error(z, label)

# Instantiate the trainer object to drive the model training

# learning_rate = 0.5
# lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
# learner = sgd(z.parameters, lr_schedule)

# --cntk --cntk_learning sgd|...|... -l  (--momentum 0.5)

## sgd ... cntk_vw parameter

# trainer = Trainer(z, loss, eval_error, [learner])


# learner.save_model("c:\\temp\\cntk\\model.1")



# binary cross entropy
