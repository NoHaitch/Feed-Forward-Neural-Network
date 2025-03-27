from graphviz import Digraph
from src.model.matrix import Matrix


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(
        format="svg", graph_attr={"rankdir": "LR"}
    )  # LR = left to right graph

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def draw_ffnn(ffnn):
    """Draws the FFNN as a feedforward neural network, using labels from the model."""
    dot = Digraph(
        format="svg", graph_attr={"rankdir": "LR", "nodesep": "1", "ranksep": "4"}
    )

    # Input Layer
    input_layer_size = ffnn.X.shape[1]
    with dot.subgraph() as sub:
        sub.attr(rank="same")
        for i in range(input_layer_size):
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
                label=layer.label + "_Bias",
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

                # Connect bias to neurons
                dot.edge(
                    bias_node_name,
                    neuron_name,
                    penwidth="1.2",
                    label=f"{neuron.b.label} {neuron.b.data:.2f}",
                    tailport="e",
                    headport="w",
                )

                # Connect neurons from previous layer
                if layer_idx == 0:
                    for i in range(input_layer_size):
                        input_node = f"Input{i+1}"
                        weight = neuron.w[i]
                        dot.edge(
                            input_node,
                            neuron_name,
                            penwidth="1.2",
                            label=f"{weight.label} {weight.data:.2f}",
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
                            label=f"{weight.label} {weight.data:.2f}",
                            tailport="e",
                            headport="w",
                        )

    return dot
