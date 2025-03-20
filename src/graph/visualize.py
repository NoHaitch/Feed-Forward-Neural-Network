from graphviz import Digraph


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
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

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
    """ Draws the FFNN as a feedforward neural network, including the input layer."""
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR", "nodesep": "1.5"})  # Increase node spacing
    
    # Input Layer
    input_layer_size = ffnn.X.cols
    with dot.subgraph() as sub:
        sub.attr(rank="same")
        for i in range(input_layer_size):
            node_name = f"Input{i}"
            sub.node(node_name, label=f"Input {i}", shape="circle", width="0.6")  # Adjust width
    
    # Hidden and Output Layers
    for layer_idx, layer in enumerate(ffnn.NN.layers):
        with dot.subgraph() as sub:
            sub.attr(rank="same")  # Ensure neurons in a layer are aligned horizontally
            for neuron_idx, neuron in enumerate(layer.neurons):
                node_name = f"L{layer_idx}N{neuron_idx}"
                sub.node(node_name, label=f"Neuron {neuron_idx}\n{neuron.active_func.__name__}", shape="circle", width="0.7")
                
                # Connect input layer to first hidden layer
                if layer_idx == 0:
                    for i in range(input_layer_size):
                        input_node = f"Input{i}"
                        dot.edge(input_node, node_name, penwidth="1.2")
                else:
                    prev_layer_size = len(ffnn.NN.layers[layer_idx - 1].neurons)
                    for prev_neuron_idx in range(prev_layer_size):
                        prev_node_name = f"L{layer_idx - 1}N{prev_neuron_idx}"
                        dot.edge(prev_node_name, node_name, penwidth="1.2")  # Connect previous layer neurons
    
    return dot