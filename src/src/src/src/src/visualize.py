import networkx as nx
from pyvis.network import Network

def export_interactive_graph(graph: nx.DiGraph, output_file: str = "interactive_knowledge_graph.html"):
    print(f"Generating interactive graph visualization -> {output_file}...")
    net = Network(height="800px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)
    
    color_map = {
        "PERSON": "#00d2ff",
        "TEAM": "#ff9f43",
        "PROJECT": "#2ecc71",
        "ACTION_ITEM": "#ff5252",
        "CONCEPT": "#9b59b6"
    }

    for node, data in graph.nodes(data=True):
        label = data.get("label", "CONCEPT")
        color = color_map.get(label, "#a4b0be")
        
        title = f"Type: {label}\n"
        for k, v in data.items():
            if k != "label":
                title += f"{k}: {v}\n"
        
        net.add_node(node, label=str(node), title=title, color=color, shape="dot", size=25 if label == "TEAM" else 15)
    
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation", "")
        net.add_edge(source, target, title=relation, label=relation, color="#8395a7", arrows="to")

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.015,
          "springLength": 120,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)
    
    net.write_html(output_file)
    print("Interactive graph saved successfully!")
