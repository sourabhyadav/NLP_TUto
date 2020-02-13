import numpy as np

timesteps = 100
input_feature = 32
output_features = 64

inputs = np.random.random((timesteps, input_feature))
print("input shape: ", inputs.shape)

state_t = np.zeros((output_features,))
print("state_t shape: ", state_t.shape)

W = np.random.random((output_features, input_feature))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

succisive_outputs = []

for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    succisive_outputs.append(output_t)

    state_t = output_t

print("before concat succisive_outputs shape: ", len(succisive_outputs))
final_output_sequence = np.concatenate(succisive_outputs, axis= 0)
print("final out seq: ", final_output_sequence)
print("final out seq shape: ", final_output_sequence.shape)