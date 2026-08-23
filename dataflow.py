from graphviz import Digraph

# Context Diagram
context_diagram = Digraph("Context Diagram", format="png")
context_diagram.attr(rankdir="LR", size="8,5")

# Nodes
context_diagram.node("User", shape="ellipse", color="lightblue", style="filled", fontsize="14")
context_diagram.node("System", shape="box", color="lightgreen", style="filled", fontsize="14")

# Edges
context_diagram.edge("User", "System", label="Upload Image", fontsize="12")
context_diagram.edge("System", "User", label="Class Label & Probabilities", fontsize="12")

# Save Context Diagram
context_diagram.render("context_diagram", view=True)

# Level 1 DFD
level1_dfd = Digraph("Level 1 DFD", format="png")
level1_dfd.attr(rankdir="TB", size="10,8")

# Nodes
level1_dfd.node("Input", "Input Image", shape="ellipse", color="lightblue", style="filled", fontsize="14")
level1_dfd.node("Preprocessing", "Data Preprocessing", shape="box", color="lightgreen", style="filled", fontsize="14")
level1_dfd.node("Feature", "Feature Extraction\n(e.g., ResNet50)", shape="box", color="lightgreen", style="filled", fontsize="14")
level1_dfd.node("Classification", "Classification\n(Dense Layers)", shape="box", color="lightgreen", style="filled", fontsize="14")
level1_dfd.node("Output", "Class Label\n& Probabilities", shape="ellipse", color="lightblue", style="filled", fontsize="14")

# Edges
level1_dfd.edge("Input", "Preprocessing", label="Input Image", fontsize="12")
level1_dfd.edge("Preprocessing", "Feature", label="Preprocessed Image", fontsize="12")
level1_dfd.edge("Feature", "Classification", label="Extracted Features", fontsize="12")
level1_dfd.edge("Classification", "Output", label="Predicted Class & Probability", fontsize="12")

# Save Level 1 DFD
level1_dfd.render("level1_dfd", view=True)
