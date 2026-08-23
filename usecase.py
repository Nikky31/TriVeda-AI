from graphviz import Digraph

# Use Case Diagram
use_case = Digraph("Use Case Diagram", format="png")
use_case.attr(rankdir="LR", size="10,8")

# Actors
use_case.node("User", shape="ellipse", color="lightblue", style="filled", fontsize="14")
use_case.node("Admin", "System Administrator", shape="ellipse", color="lightpink", style="filled", fontsize="14")

# System Boundary
use_case.node("System", shape="box", color="black", style="rounded", width="2", height="3")

# Use Cases
use_case.node("Upload", "Upload Image", shape="ellipse", color="lightgreen")
use_case.node("Process", "Process Image", shape="ellipse", color="lightgreen")
use_case.node("View", "View Results", shape="ellipse", color="lightgreen")
use_case.node("Update", "Update Model", shape="ellipse", color="lightgreen")

# Relationships
use_case.edge("User", "Upload")
use_case.edge("User", "View")
use_case.edge("Admin", "Update")
use_case.edge("System", "Upload")
use_case.edge("System", "Process")
use_case.edge("System", "View")
use_case.edge("System", "Update")

# Save Diagram
use_case.render("use_case_diagram", view=True)
