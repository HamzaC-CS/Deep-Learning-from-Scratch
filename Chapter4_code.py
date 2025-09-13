import numpy as np

from lincoln import activations
from lincoln import layers
from lincoln import losses
from lincoln import optimizers
from lincoln import network
from lincoln import train
from lincoln.utils import mnist

RANDOM_SEED = 190119

# one-hot encode
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Sigmoid()),
    ],
    loss=losses.MeanSquaredError(normalize=False),
    seed=RANDOM_SEED,
)

trainer = train.Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)

model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Sigmoid()),
    ],
    loss=losses.MeanSquaredError(normalize=True),
    seed=RANDOM_SEED,
)

trainer = train.Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)

model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

trainer = train.Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)
print()
calc_accuracy_model(model, X_test)

model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

optim = optimizers.SGDMomentum(0.1, momentum=0.9)

trainer = train.Trainer(model, optim)
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)

# Different weight decay 
model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

optimizer = optimizers.SGDMomentum(0.15, momentum=0.9, final_lr=0.05, decay_type='linear')

trainer = train.Trainer(model, optimizer)
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=25,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)

model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

optimizer = optimizers.SGDMomentum(0.2, momentum=0.9, final_lr=0.05, decay_type='exponential')

trainer = train.Trainer(model, optimizer)
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=25,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid()),
        layers.Dense(neurons=10, activation=activations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

optimizer = optimizers.SGDMomentum(0.2, momentum=0.9, final_lr=0.05, decay_type='exponential')

trainer = train.Trainer(model, optimizer)
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=25,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)

# Changing weight init
model = network.NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=activations.Sigmoid(), weight_init="glorot"),
        layers.Dense(neurons=10, activation=activations.Linear(), weight_init="glorot"),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)

optimizer = optimizers.SGDMomentum(0.2, momentum=0.9, final_lr=0.05, decay_type='exponential')

trainer = train.Trainer(model, optimizer)
trainer.fit(
    X_train,
    train_labels,
    X_test,
    test_labels,
    epochs=25,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)

calc_accuracy_model(model, X_test)






