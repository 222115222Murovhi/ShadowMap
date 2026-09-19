class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):

        output = input_data

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, output_gradient):

        gradient = output_gradient

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def get_parameters(self):

        """
        Collect parameters from all trainable layers.

        Layers such as ReLU are not trainable, so they do
        not have weights or biases to save.
        """

        parameters = []

        for layer in self.layers:

            if layer.trainable:

                parameters.append(
                    layer.get_parameters()
                )

        return parameters

    def set_parameters(self, parameters):

        """
        Restore parameters to all trainable layers.

        The parameters are assigned in the same order in
        which they were originally collected.
        """

        parameter_index = 0

        for layer in self.layers:

            if layer.trainable:

                layer.set_parameters(
                    parameters[parameter_index]
                )

                parameter_index += 1

    def set_training(self, training):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training