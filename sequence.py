from graphviz import Digraph
sequence_diagram = Digraph("Sequence Diagram", format="png")
sequence_diagram.attr(rankdir="TB", size="8,10")

# Nodes
sequence_diagram.node("User", shape="rect", style="filled", color="lightblue")
sequence_diagram.node("System", shape="rect", style="filled", color="lightgreen")
sequence_diagram.node("Database", shape="rect", style="filled", color="lightyellow")

# Messages
sequence_diagram.edge("User", "System", label="Upload Image")
sequence_diagram.edge("System", "System", label="Preprocess Image")
sequence_diagram.edge("System", "System", label="Extract Features & Classify")
sequence_diagram.edge("System", "Database", label="Store Results")
sequence_diagram.edge("System", "User", label="Return Predicted Label & Confidence")

# Save Diagram
sequence_diagram.render("sequence_diagram", view=True)
