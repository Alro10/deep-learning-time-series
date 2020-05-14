## LSTM questions:

- How to define the input tensor and what this input must be 3D?

```Python
model = Sequential()
model.add(LSTM(input_shape = (50,1), output_dim= 50, return_sequences = True))
```
    - time_step = # of neurons = 50
    - hidden_states = output_dim = 50
    - real input_shape = (X_train, 50, 1) >> output (512, 50, 50) each epoch = (batch_size, time_step, unit)

- How the dimension of vector changes along hidden layers?

- What is `dropout` exactly?

- How to define the output tensor correctly?

### References

- https://adventuresinmachinelearning.com/keras-lstm-tutorial/

- https://stackoverflow.com/questions/49892528/what-is-the-architecture-behind-the-keras-lstm-layer-implementation

- https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras

- https://stackoverflow.com/questions/44273249/in-keras-what-exactly-am-i-configuring-when-i-create-a-stateful-lstm-layer-wi
