from graphviz import Digraph
import matplotlib.pyplot as plt

class Visualizer:
    """A static class for visualizing neural networks and their components."""

    @staticmethod
    def trace(root):
        """Builds a set of all nodes and edges in a computation graph."""
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    @staticmethod
    def draw_dot(root):
        """Draws a computation graph using Graphviz."""
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        nodes, edges = Visualizer.trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot

    @staticmethod
    def draw_ffnn(ffnn, input_size):
        """
        Draws the FFNN structure using Graphviz, keeping weight labels and gradients on edges.

        Args:
            ffnn (FFNN): The feedforward neural network model.
            input_size (int): Number of input features.
        """
        dot = Digraph(
            format="svg", graph_attr={"rankdir": "LR", "nodesep": "1", "ranksep": "4"}
        )

        # Input Layer
        with dot.subgraph() as sub:
            sub.attr(rank="same")
            for i in range(input_size):
                node_name = f"Input{i+1}"
                sub.node(node_name, label=f"Input {i+1}", shape="circle", width="0.6")

        # Hidden and Output Layers
        for layer_idx, layer in enumerate(ffnn.NN.layers):
            with dot.subgraph() as sub:
                sub.attr(rank="same")

                # Bias Node
                bias_node_name = f"Bias_L{layer_idx+1}"
                dot.node(
                    bias_node_name,
                    label=f"{layer.label}_Bias",
                    shape="circle",
                    width="0.5",
                )

                for neuron in layer.neurons:
                    neuron_name = neuron.label
                    sub.node(
                        neuron_name,
                        label=f"{neuron.label} | {neuron.active_func.__name__}",
                        shape="circle",
                        width="0.7",
                    )

                    # Connect bias to neurons (with label)
                    dot.edge(
                        bias_node_name,
                        neuron_name,
                        penwidth="1.2",
                        label=f"{neuron.b.label}\nw: {neuron.b.data:.2f}\ngrad: {neuron.b.grad:.4f}",
                        tailport="e",
                        headport="w",
                    )

                    # Connect neurons from previous layer (with weight labels)
                    if layer_idx == 0:
                        for i in range(input_size):
                            input_node = f"Input{i+1}"
                            weight = neuron.w[i]
                            dot.edge(
                                input_node,
                                neuron_name,
                                penwidth="1.2",
                                label=f"{weight.label}\nw: {weight.data:.2f}\ngrad: {weight.grad:.4f}",
                                tailport="e",
                                headport="w",
                            )
                    else:
                        prev_layer = ffnn.NN.layers[layer_idx - 1]
                        for prev_neuron in prev_layer.neurons:
                            weight = neuron.w[prev_layer.neurons.index(prev_neuron)]
                            dot.edge(
                                prev_neuron.label,
                                neuron_name,
                                penwidth="1.2",
                                label=f"{weight.label}\nw: {weight.data:.4f}\ngrad: {weight.grad:.8f}",
                                tailport="e",
                                headport="w",
                            )

        return dot

    @staticmethod
    def plot_weight_distribution(model, layer_indices):
        """
        Plot the distribution of weights in specified layers.

        Args:
            model (MLP): The neural network model.
            layer_indices (list[int]): List of layer indices to plot.

        Raises:
            ValueError: If a layer index does not exist.
        """
        num_layers = len(model.layers)

        for layer_idx in layer_indices:
            if layer_idx >= num_layers or layer_idx < 0:
                raise ValueError(
                    f"Layer index {layer_idx} is out of range. Model has {num_layers} layers."
                )

            layer = model.layers[layer_idx]
            weights = [w.data for neuron in layer.neurons for w in neuron.w]

            if not weights:
                print(f"Warning: No weights found in layer {layer_idx}")
                continue

            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=20, alpha=0.75, edgecolor="black")
            plt.title(f"Weight Distribution - Layer {layer_idx} ({layer.label})")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

    @staticmethod
    def plot_gradient_distribution(model, layer_indices):
        """
        Plot the distribution of gradients of weights in specified layers.

        Args:
            model (MLP): The neural network model.
            layer_indices (list[int]): List of layer indices to plot.

        Raises:
            ValueError: If a layer index does not exist.
        """
        num_layers = len(model.layers)

        for layer_idx in layer_indices:
            if layer_idx >= num_layers or layer_idx < 0:
                raise ValueError(
                    f"Layer index {layer_idx} is out of range. Model has {num_layers} layers."
                )

            layer = model.layers[layer_idx]
            gradients = [w.grad for neuron in layer.neurons for w in neuron.w]

            if not gradients:
                print(f"Warning: No gradients found in layer {layer_idx}")
                continue

            plt.figure(figsize=(6, 4))
            plt.hist(gradients, bins=20, alpha=0.75, edgecolor="black", color="red")
            plt.title(f"Gradient Distribution - Layer {layer_idx} ({layer.label})")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

    @staticmethod
    def plot_loss_history(loss_history):
        """
        Plots the training and validation loss history in two separate graphs.

        Args:
            loss_history (dict): A dictionary containing:
                - "train_loss": List of training losses per batch.
                - "val_loss": List of validation losses per epoch.
        """
        # Plot Training Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_history["train_loss"]) + 1), loss_history["train_loss"], 
                 label="Training Loss", color="blue", marker="o")
        plt.xlabel("Steps (Batches)")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_history["val_loss"]) + 1), loss_history["val_loss"], 
                 label="Validation Loss", color="red", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def compare_loss_histories(history_list):
        """
        Compares multiple loss histories in two side-by-side plots.

        Args:
            history_list (list of dict): Each dictionary should contain:
                - "train_loss" (list of float): Training loss per batch.
                - "val_loss" (list of float): Validation loss per epoch.
        
        Example:
            history1 = {"train_loss": [..], "val_loss": [..]}
            history2 = {"train_loss": [..], "val_loss": [..]}
            compare_loss_histories([history1, history2])
        """
        plt.figure(figsize=(12, 5))

        # Plot Training Loss (Smoothed per Epoch)
        plt.subplot(1, 2, 1)
        for history in history_list:
            train_loss_per_epoch = [history["train_loss"][i] for i in range(len(history["val_loss"]))] 
            epochs = range(1, len(train_loss_per_epoch) + 1)
            plt.plot(epochs, train_loss_per_epoch, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison")
        plt.grid(True)

        # Plot Validation Loss
        plt.subplot(1, 2, 2)
        for history in history_list:
            epochs = range(1, len(history["val_loss"]) + 1)
            plt.plot(epochs, history["val_loss"], marker='s')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Comparison")
        plt.grid(True)

        plt.tight_layout()
        plt.show()