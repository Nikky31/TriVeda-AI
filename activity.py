from graphviz import Digraph
activity_diagram = Digraph("Activity Diagram", format="png")
activity_diagram.attr(rankdir="TB", size="10,8")

# Nodes
activity_diagram.node("Start", shape="ellipse", label="Start", color="green")
activity_diagram.node("Upload", shape="box", label="Upload Image", color="lightblue", style="filled")
activity_diagram.node("Preprocess", shape="box", label="Preprocess Image", color="lightgreen", style="filled")
activity_diagram.node("Classify", shape="box", label="Classify Image", color="lightgreen", style="filled")
activity_diagram.node("Return", shape="box", label="Return Results", color="lightblue", style="filled")
activity_diagram.node("End", shape="ellipse", label="End", color="red")

# Edges
activity_diagram.edge("Start", "Upload")
activity_diagram.edge("Upload", "Preprocess")
activity_diagram.edge("Preprocess", "Classify")
activity_diagram.edge("Classify", "Return")
activity_diagram.edge("Return", "End")

# Save Diagram
activity_diagram.render("activity_diagram", view=True)
