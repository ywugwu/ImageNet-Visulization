import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import collections

# use wide layout
st.set_page_config(layout="wide")

# Add this at the beginning of your script, after the imports
st.sidebar.title("Visualization Settings")
THRESHOLD = st.sidebar.selectbox(
    "Select Max Number of Nodes to Display",
    options=["128", "256", "512", "1024", "No limit"],
    index=0  # Default to 128
)
THRESHOLD = None if THRESHOLD == "No limit" else int(THRESHOLD)

# Add new selectbox for weight standardization
STANDARDIZE_WEIGHTS = st.sidebar.selectbox(
    "Standardize Weights",
    options=["No", "Yes"],
    index=0  # Default to No
)
STANDARDIZE_WEIGHTS = STANDARDIZE_WEIGHTS == "Yes"

# set title
st.title("ImageNet Tree Visualization")

@st.cache_data
def load_data():
    with open('imageNet_text_weights.json') as f:
        weights = json.load(f)
    weights = {name.lower(): score['type_2_text_consistency_score'] for name, score in weights.items()}
    
    with open('imagenet_tree.json') as f:
        tree_data = json.load(f)
    
    return weights, tree_data

weights, tree_data = load_data()

# Add function to standardize weights
def standardize_weights(weights):
    values = list(weights.values())
    min_val = min(values)
    max_val = max(values)
    return {k: (v - min_val) / (max_val - min_val) for k, v in weights.items()}

# Apply standardization if selected
if STANDARDIZE_WEIGHTS:
    weights = standardize_weights(weights)

class Node:
    def __init__(self, id, label, shape, size, value):
        self.id = id
        self.label = label
        self.shape = shape
        self.size = size
        self.value = value  # Add value attribute

class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target

def build_graph(data, threshold):
    nodes = []
    edges = []
    vis = {}
    values_at_depth = collections.defaultdict(list)
    
    def dfs(node, depth):
        if node['name'].lower() in weights:
            values_at_depth[depth].append(weights[node['name'].lower()])
        if threshold is not None and len(vis) >= threshold:  # Limit the number of nodes for clarity
            return
        if node['id'] not in vis:
            vis[node['id']] = True
            nodes.append(
                Node(
                    id=node['id'],
                    label=node['name'].split(',')[0],
                    shape="dot",
                    size=25,
                    value=weights.get(node['name'].lower(), -1.0)  # Get value from weights dictionary
                )
            )
        if 'children' in node:
            for child in node['children']:
                edges.append(
                    Edge(
                        source=node['id'],
                        target=child['id'],
                    )
                )
                dfs(child, depth+1)
    
    dfs(data, 0)
    return nodes, edges, values_at_depth

nodes, edges, values_at_depth = build_graph(tree_data, THRESHOLD)

st.header(f"Please Wait for the Rendering of {len(nodes)} nodes and {len(edges)} edges to Complete:coffee:...")
st.header(f"Green = High Score, Red = Low Score, Gray = N/A")
st.header(f"Weights {'are' if STANDARDIZE_WEIGHTS else 'are not'} standardized")

# The rest of your visualization code here...
def get_color(value):
    if value == -1:
        return '#808080'  # Gray for undefined values
    elif value >= 0:
        # Mapping from 0 to 1 (Red to Green in RGB)
        return mcolors.rgb2hex((1 - value, value, 0))
    else:
        # Default to gray if something unexpected
        return '#808080'

# Generate colors using the custom function
values = [node.value for node in nodes]
colors = [get_color(value) for value in values]

# Convert Python objects to JavaScript-readable format
nodes_js = [{'id': n.id, 'label': n.label, 'shape': n.shape, 'size': n.size, 'value': n.value, 'color': color} 
            for n, color in zip(nodes, colors)]
edges_js = [{'from': e.source, 'to': e.target} for e in edges]

html_string = f"""
<html>
<head>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div id="mynetwork" style="height: 800px; width: 100%;"></div>
  <script type="text/javascript">
    var nodes = new vis.DataSet({json.dumps(nodes_js)});
    var edges = new vis.DataSet({json.dumps(edges_js)});
    var container = document.getElementById('mynetwork');
    var data = {{
        nodes: nodes,
        edges: edges
    }};
    var options = {{
        layout: {{
            hierarchical: {{
                enabled: true,
                direction: 'LR',  // Left to Right
                sortMethod: 'hubsize' //'directed'  // Direct sorting to maintain direction
            }}
        }},
        nodes: {{
            shape: 'dot',
            font: {{
                size: 15
            }}
        }},
        edges: {{
            width: 2
        }}
    }};
    var network = new vis.Network(container, data, options);
  </script>
</body>
</html>
"""

st.components.v1.html(html_string, height=800)

# Optional: Add the scatter plot
# fig, ax = plt.subplots()
# for depth, values in values_at_depth.items():
#     ax.scatter([depth] * len(values), values, c='b', alpha=0.5)
# ax.set_xlabel('Depth')
# ax.set_ylabel('Value')
# ax.set_title('Consistency Score Distribution at Each Depth')
# st.pyplot(fig)