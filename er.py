from graphviz import Digraph

# Initialize ERD
erd = Digraph("ERD", format="png")
erd.attr(rankdir="LR", size="10,8")

# Entities
erd.node("ImageDataset", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightblue"><B>ImageDataset</B></TD></TR>
  <TR><TD>Image_ID (PK)</TD></TR>
  <TR><TD>Image_Path</TD></TR>
  <TR><TD>Class_Label</TD></TR>
</TABLE>>''', shape="plaintext")

erd.node("Model", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightgreen"><B>Model</B></TD></TR>
  <TR><TD>Model_ID (PK)</TD></TR>
  <TR><TD>Model_Type</TD></TR>
  <TR><TD>Pretrained_Weights</TD></TR>
  <TR><TD>Accuracy</TD></TR>
</TABLE>>''', shape="plaintext")

erd.node("Results", '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR><TD BGCOLOR="lightyellow"><B>Results</B></TD></TR>
  <TR><TD>Result_ID (PK)</TD></TR>
  <TR><TD>Image_ID (FK)</TD></TR>
  <TR><TD>Predicted_Label</TD></TR>
  <TR><TD>Confidence_Score</TD></TR>
</TABLE>>''', shape="plaintext")

# Relationships
erd.edge("ImageDataset", "Results", label="Has", fontsize="12")
erd.edge("Model", "Results", label="Generates", fontsize="12")

# Save ERD
erd.render("erd", view=True)
