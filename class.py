from graphviz import Digraph
class_diagram = Digraph("Class Diagram", format="png")
class_diagram.attr(rankdir="TB", size="8,10")

# Classes
class_diagram.node("Image", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightblue"><B>Image</B></TD></TR>
  <TR><TD>Image_ID</TD></TR>
  <TR><TD>Path</TD></TR>
  <TR><TD>Label</TD></TR>
</TABLE>>''', shape="plaintext")

class_diagram.node("Model", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightgreen"><B>Model</B></TD></TR>
  <TR><TD>Model_Name</TD></TR>
  <TR><TD>Accuracy</TD></TR>
  <TR><TD>Loss</TD></TR>
</TABLE>>''', shape="plaintext")

class_diagram.node("Classifier", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightyellow"><B>Classifier</B></TD></TR>
  <TR><TD>Features</TD></TR>
  <TR><TD>Predicted_Label</TD></TR>
  <TR><TD>Confidence</TD></TR>
</TABLE>>''', shape="plaintext")

# Relationships
class_diagram.edge("Image", "Classifier", label="Processes")
class_diagram.edge("Model", "Classifier", label="Utilized by")

# Save Diagram
class_diagram.render("class_diagram", view=True)
