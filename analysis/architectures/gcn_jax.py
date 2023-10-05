'''
Graph Convolutional Network using JAX via jraph+flax

This class defines the JAX architecture using flax: setup(), __call__()

The GNN should be constructed from a module object (which can contain submodules)

Parts of this are based on:
 - https://github.com/deepmind/jraph
 - https://colab.research.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
 - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
'''

import jax
import jax.numpy as jnp
import flax
import jraph

##########################################
class GCN(flax.linen.Module):
    '''
    :param hidden_dims: list of hidden dimensions
    :param n_output_classes: number of output classes
    '''
    hidden_dim : int
    n_output_classes : int

    #----------------------------------------
    def setup(self):
        self.gcn = jraph.GraphConvolution(update_node_fn=self.node_update_fn,
                                          aggregate_nodes_fn = jraph.segment_sum,
                                          add_self_edges = True,
                                          symmetric_normalization = True)
        self.mlp = MLP(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.output_layer = MLP(hidden_dim=self.hidden_dim, output_dim=self.n_output_classes)

    #----------------------------------------
    # Define node update function
    def node_update_fn(self, node_features: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(node_features)

    #----------------------------------------
    def __call__(self, graph):
        '''
        This function defines the forward pass of the network
        '''
        # GCN layer
        graph = self.gcn(graph)

        # Create graph representation by averaging node features for each graph in the batch
        mean_node_features = jnp.sum(graph.nodes, axis=1) / jnp.expand_dims(graph.n_node, axis=-1)

        # Return logits
        logits = self.output_layer(mean_node_features)
        return logits

##########################################
class MLP(flax.linen.Module):
    '''
    We define a simple MLP w/flax to be used as a component of our GNN.
    '''
    hidden_dim : int
    output_dim : int

    #----------------------------------------
    @flax.linen.compact
    def __call__(self, x):
        '''
        This function defines the forward pass of the network
        '''
        x = flax.linen.Dense(features=self.hidden_dim)(x)
        x = jax.nn.relu(x)
        x = flax.linen.Dense(features=self.output_dim)(x)
        return x